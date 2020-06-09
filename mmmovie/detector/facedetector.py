import json
import os.path as osp

import mmcv
import torch
from mmcv.runner import load_checkpoint

from .facedet.dataset import FaceDataProcessor
from .facedet.mtcnn import MTCNN


class FaceDetector(object):

    def __init__(self, cfg_path, weight_path, gpu=0, img_scale=(1333, 800)):
        self.model = self.build_mtcnn(cfg_path, weight_path)
        self.model.eval()
        self.model.cuda(gpu)
        self.data_processor = FaceDataProcessor(gpu)

    def build_mtcnn(self, cfg_path, weight_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        model = MTCNN(**cfg)
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
            results = self.model(data)
        bboxes = results[0][0]
        landmarks = results[1][0]
        keep_idx = bboxes[:, -1] > conf_thr
        bboxes = bboxes[keep_idx]
        landmarks = landmarks[keep_idx]
        return bboxes, landmarks
