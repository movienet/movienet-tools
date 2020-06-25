import json
import os.path as osp
import shutil
import tempfile
from functools import partial

import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from torch.utils.data import DataLoader

from .persondet.cascade_rcnn import CascadeRCNN
from .persondet.dataset import CustomDataset, DataProcessor
from .persondet.retinanet import RetinaNet

from mmmovie.utils import PaddedSampler, acc_list


class ParallelPersonDetector(object):

    def __init__(self,
                 arch,
                 cfg_path,
                 weight_path,
                 gpu_ids=None,
                 img_scale=(1333, 800)):
        # build model
        if arch == 'retina':
            self.model = self.build_retinanet(cfg_path, weight_path)
        elif arch == 'rcnn':
            self.model = self.build_cascadercnn(cfg_path, weight_path)
        else:
            raise KeyError('{} is not supported now.'.format(arch))
        if gpu_ids is None:
            self.gpu_ids = list(range(torch.cuda.device_count()))
        else:
            self.gpu_ids = gpu_ids
        assert isinstance(self.gpu_ids, list)
        self.ngpu = len(self.gpu_ids)
        self.model = MMDataParallel(
            self.model.cuda(self.gpu_ids[0]), device_ids=self.gpu_ids)
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
        # build data loader
        sampler = PaddedSampler(dataset, imgs_per_gpu, self.ngpu)
        data_loader = DataLoader(
            dataset,
            batch_size=imgs_per_gpu * self.ngpu,
            sampler=sampler,
            num_workers=workers_per_gpu * self.ngpu,
            collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
            pin_memory=False)
        results = self.multi_gpu_test(data_loader, conf_thr)
        results = results[:len(dataset)]
        return results

    def multi_gpu_test(self, data_loader, conf_thr):
        bboxes, nboxes = [], []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))

        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = self.model(return_numpy=False, rescale=True, **data)
            bboxes.append(result['bbox'].detach().cpu().numpy())
            nboxes.append(result['nbox'].detach().cpu().numpy())
            # result = result[result[:, -1] > conf_thr]
            # results.append(result)
            for _ in range(self.ngpu):
                prog_bar.update()

        # unpack result
        bboxes = np.concatenate(bboxes, axis=0)
        nboxes = acc_list(np.concatenate(nboxes, axis=0))[:-1]
        result_bbox = np.split(bboxes, nboxes, axis=0)

        # post process
        result_bbox = result_bbox[:len(data_loader.dataset)]
        result_bbox = [r[r[:, -1] > conf_thr] for r in result_bbox]

        return result_bbox
