import json
import os.path as osp
import shutil
import tempfile
from functools import partial

import cv2
import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDistributedDataParallel, collate
from mmcv.runner import get_dist_info, load_checkpoint
from torch.utils.data import DataLoader, DistributedSampler

from .persondet.cascade_rcnn import CascadeRCNN
from .persondet.dataset import CustomDataset, DataProcessor
from .persondet.retinanet import RetinaNet

resources_dir = osp.join(osp.dirname(__file__), '../../../model')


class PersonDetector(object):

    def __init__(self,
                 arch='rcnn',
                 cfg_path=osp.join(resources_dir,
                                   'cascade_rcnn_x101_64x4d_fpn.json'),
                 weight_path=osp.join(resources_dir,
                                      'cascade_rcnn_x101_64x4d_fpn.pth'),
                 gpu=0,
                 img_scale=(1333, 800)):
        if arch == 'retina':
            self.model = self.build_retinanet(cfg_path, weight_path)
        elif arch == 'rcnn':
            self.model = self.build_cascadercnn(cfg_path, weight_path)
        else:
            raise KeyError('{} is not supported now.'.format(arch))
        self.model.eval()
        self.model.cuda(gpu)
        self.data_processor = DataProcessor(gpu, img_scale)

    def build_retinanet(self, cfg_path, weight_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        model = RetinaNet(**cfg)
        load_checkpoint(model, weight_path, map_location='cpu')
        return model

    def build_cascadercnn(self, cfg_path, weight_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        model = CascadeRCNN(**cfg)
        load_checkpoint(model, weight_path, map_location='cpu')
        return model

    def detect(self, img, conf_thr=0.5, show=False):
        assert conf_thr >= 0 and conf_thr < 1
        if isinstance(img, str):
            filename = img
            assert osp.isfile(filename)
            img = mmcv.imread(filename)
        data = self.data_processor(img)
        with torch.no_grad():
            persons = self.model(rescale=True, **data)
        persons = persons[persons[:, -1] > conf_thr]
        if show:
            img_show = self.draw_person(img.copy(), persons)
            cv2.imshow('person', img_show)
            cv2.waitKey()
        return persons

    def draw_person(self, img, persons):
        num_persons = persons.shape[0]
        for i in range(num_persons):
            img = cv2.rectangle(img, (int(persons[i, 0]), int(persons[i, 1])),
                                (int(persons[i, 2]), int(persons[i, 3])),
                                (0, 255, 0), 2)
            img = cv2.putText(
                img, 'prob:{:.2f}'.format(persons[i, -1]),
                (int(persons[i, 0] + 10), int(persons[i, 1] + 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return img

    def crop_person(self,
                    img,
                    persons,
                    save_dir=None,
                    save_prefix=None,
                    img_scale=(128, 256)):
        num_persons = persons.shape[0]
        h, w = img.shape[:2]
        person_list = []
        for i in range(num_persons):
            x1, y1, x2, y2 = persons[i, :4].astype(np.int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            person_img = img[y1:y2, x1:x2]
            if img_scale is not None:
                person_img = cv2.resize(person_img, img_scale)
            if save_dir is not None and save_prefix is not None:
                save_path = osp.join(save_dir,
                                     save_prefix + '_{:02d}.jpg'.format(i))
                cv2.imwrite(save_path, person_img)
            person_list.append(person_img)
        return person_list


class DistPersonDetector(object):

    def __init__(self, arch, cfg_path, weight_path, img_scale=(1333, 800)):
        # build model
        if arch == 'retina':
            self.model = self.build_retinanet(cfg_path, weight_path)
        elif arch == 'rcnn':
            self.model = self.build_cascadercnn(cfg_path, weight_path)
        else:
            raise KeyError('{} is not supported now.'.format(arch))
        self.model = MMDistributedDataParallel(
            self.model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        self.model.eval()
        self.img_scale = img_scale

    def build_retinanet(self, cfg_path, weight_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        model = RetinaNet(**cfg)
        load_checkpoint(model, weight_path, map_location='cpu')
        return model

    def build_cascadercnn(self, cfg_path, weight_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        model = CascadeRCNN(**cfg)
        load_checkpoint(model, weight_path, map_location='cpu')
        return model

    def batch_detect(self,
                     imglist,
                     img_prefix,
                     imgs_per_gpu=1,
                     workers_per_gpu=4,
                     conf_thr=0.5):
        # build dataset
        dataset = CustomDataset(
            imglist, img_scale=self.img_scale, img_prefix=img_prefix)
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
        results = self.multi_gpu_test(data_loader, conf_thr)
        return results

    def multi_gpu_test(self, data_loader, conf_thr):
        results = []
        dataset = data_loader.dataset
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = self.model(rescale=True, **data)
            result = result[result[:, -1] > conf_thr]
            results.append(result)
            if rank == 0:
                batch_size = (
                    len(data['img_meta']._data)
                    if 'img_meta' in data else data['img'].size(0))
                for _ in range(batch_size * world_size):
                    prog_bar.update()
        # collect results from all ranks
        results = self.collect_results(results, len(dataset))
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
        mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
        dist.barrier()
        # collect all parts
        if rank != 0:
            return None
        else:
            # load results of all parts from tmp dir
            part_list = []
            for i in range(world_size):
                part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
                part_list.append(mmcv.load(part_file))
            # sort the results
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            ordered_results = ordered_results[:size]
            # remove tmp dir
            shutil.rmtree(tmpdir)
            return ordered_results
