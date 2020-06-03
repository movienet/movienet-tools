import json
import os.path as osp

import mmcv
import torch
from mmcv.runner import load_checkpoint

from .persondet.cascade_rcnn import CascadeRCNN
from .persondet.dataset import DataProcessor
from .persondet.retinanet import RetinaNet


class PersonDetector(object):

    def __init__(self,
                 arch,
                 cfg_path,
                 weight_path,
                 gpu=1,
                 img_scale=(1333, 800)):
        if arch == 'retina':
            self.model = self.build_retinanet(cfg_path, weight_path)
        elif arch == 'rcnn':
            self.model = self.build_cascadercnn(cfg_path, weight_path)
        else:
            raise KeyError('{} is not supported now.'.format(arch))
        self.model.eval()
        self.model.cuda(gpu)
        self.data_processor = DataProcessor(img_scale)

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

    def detect(self, img):
        if isinstance(img, str):
            filename = img
            assert osp.isfile(filename)
            img = mmcv.imread(filename)
        data = self.data_processor(img)
        with torch.no_grad():
            result = self.model(rescale=False, **data)
        self.model.show_result(data, result)
        return result
