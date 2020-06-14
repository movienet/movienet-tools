import mmcv
from .detectors import FastRCNN
import torch
from .src import ActionDataPreprocessor
import numpy as np


class ActionExtractor(object):

    def __init__(self, config_path, weight_path, gpu=0):

        cfg = mmcv.Config.fromfile(config_path)
        self.cfg = cfg
        self.gpu = gpu
        self.model = FastRCNN(**cfg)
        self.model.eval()
        self.model.cuda(gpu)
        mmcv.runner.load_checkpoint(
            self.model, weight_path, map_location=f"cuda:{gpu}")
        self.data_preprocessor = ActionDataPreprocessor(gpu)

    def extract(self, imgs, bboxes):
        """
        imgs: sequence of image
        bboxes: static tracklet, expressed by boxes within one frame.
        """
        assert np.logical_and(bboxes <= 1, bboxes >= 0).all()
        data = self.data_preprocessor(imgs, bboxes)
        with torch.no_grad():
            scaled_bboxes, score, feature = self.model(rescale=True, **data)
        assert len(scaled_bboxes) == len(score) == len(feature)
        return [
            dict(
                bboxes=_box.cpu().numpy(),
                score=_score.cpu().numpy(),
                action_feature=_feat.cpu().numpy())
            for _box, _score, _feat in zip(scaled_bboxes, score, feature)
        ]
