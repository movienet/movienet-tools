import logging
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import torch.nn as nn

from .modules.core.misc import tensor2imgs


class BaseDetector(nn.Module):
    """Base class for detectors."""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward(self, img, img_meta, **kwargs):
        return self.simple_test(img, img_meta, **kwargs)

    def show_result(self, data, bboxes, dataset=None, score_thr=0.3):
        img_tensor = data['img']
        if isinstance(data['img_meta'], list):
            img_metas = data['img_meta']
        else:
            img_metas = data['img_meta'].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            labels = np.full(bboxes.shape[0], 0, dtype=np.int32)
            mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names='person',
                score_thr=score_thr)
