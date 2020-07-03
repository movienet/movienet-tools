import os.path as osp
import shutil
import tempfile
from functools import partial

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDistributedDataParallel, collate
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader, DistributedSampler

from .src import FaceDataProcessor, FaceDataset, IRv1_face


class FaceExtractor(object):

    def __init__(self, weight_path, gpu=0):
        weights = torch.load(weight_path, map_location='cpu')
        self.model = IRv1_face(weights)
        self.model.eval()
        self.model.cuda(gpu)
        self.data_processor = FaceDataProcessor(gpu)

    def extract(self, img):
        if isinstance(img, str):
            filename = img
            assert osp.isfile(filename)
            img = mmcv.imread(filename)
        data = self.data_processor(img)
        with torch.no_grad():
            result = self.model(data)
        feature = result.detach().cpu().numpy().squeeze()
        return feature


class DistFaceExtractor(object):

    def __init__(self, weight_path):
        # build model
        weights = torch.load(weight_path, map_location='cpu')
        self.model = IRv1_face(weights)
        self.model = MMDistributedDataParallel(
            self.model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        self.model.eval()

    def batch_extract(self,
                      imglist,
                      img_prefix,
                      imgs_per_gpu=1,
                      workers_per_gpu=4):
        # build dataset
        dataset = FaceDataset(imglist, img_prefix=img_prefix)
        # get dist info
        rank, world_size = get_dist_info()
        # build data loader
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
        data_loader = DataLoader(
            dataset,
            batch_size=imgs_per_gpu,
            sampler=sampler,
            num_workers=workers_per_gpu,
            collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
            pin_memory=False)
        results = self.multi_gpu_test(data_loader)
        return results

    def multi_gpu_test(self, data_loader):
        results = []
        dataset = data_loader.dataset
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = self.model(data)
            results.append(result.detach().cpu())
            if rank == 0:
                batch_size = data.size(0)
                for _ in range(batch_size * world_size):
                    prog_bar.update()
        # collect results from all ranks
        results = self.collect_results(torch.cat(results), len(dataset))
        return results

    def collect_results(self, result_part, size, tmpdir=None):
        rank, world_size = get_dist_info()
        # create a tmp dir if it is not specified
        if tmpdir is None:
            MAX_LEN = 512
            # 32 is whitespace
            dir_tensor = torch.full((MAX_LEN, ),
                                    32,
                                    dtype=torch.uint8,
                                    device='cuda')
            if rank == 0:
                tmpdir = tempfile.mkdtemp()
                # pylint: disable=not-callable
                tmpdir = torch.tensor(
                    bytearray(tmpdir.encode()),
                    dtype=torch.uint8,
                    device='cuda')
                # pylint: enable=not-callable
                dir_tensor[:len(tmpdir)] = tmpdir
            dist.broadcast(dir_tensor, 0)
            tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
        else:
            mmcv.mkdir_or_exist(tmpdir)
        # dump the part result to the dir
        # mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
        torch.save(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
        dist.barrier()
        # collect all parts
        if rank != 0:
            return None
        else:
            # load results of all parts from tmp dir
            part_list = []
            for i in range(world_size):
                part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
                # part_list.append(mmcv.load(part_file))
                part_list.append(torch.load(part_file))
            # sort the results
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.append(torch.stack(res))
            # the dataloader may pad some samples
            ordered_results = torch.cat(ordered_results)
            ordered_results = ordered_results[:size]
            # remove tmp dir
            shutil.rmtree(tmpdir)
            return ordered_results
