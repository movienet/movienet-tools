import os.path as osp

import mmcv
import torch

from .src import DataProcessor, resnet50_place


class PlaceExtractor(object):

    def __init__(self, weight_path, gpu=0):
        weights = torch.load(weight_path, map_location='cpu')
        self.model = resnet50_place(weights)
        self.model.eval()
        self.model.cuda(gpu)
        self.data_processor = DataProcessor(gpu)

    def extract(self, img):
        if isinstance(img, str):
            filename = img
            assert osp.isfile(filename)
            img = mmcv.imread(filename)
        data = self.data_processor(img)
        with torch.no_grad():
            result = self.model(data)
        return result
