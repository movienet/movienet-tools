import logging
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import torch.nn as nn


class BaseDetector(nn.Module):
    """Base class for detectors"""

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

    @abstractmethod
    def extract_feat(self, img_group):
        pass

    def extract_feats(self, img_groups):
        assert isinstance(img_groups, list)
        for img_group in img_groups:
            yield self.extract_feat(img_group)

    @abstractmethod
    def forward_train(self, num_modalities, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, num_modalities, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, num_modalities, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, num_modalities, img_metas, **kwargs):
        if not isinstance(img_metas, list):
            raise TypeError('{} must be a list, but got {}'.format(
                img_metas, type(img_metas)))

        num_augs = len(kwargs['img_group_0'])
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    num_augs, len(img_metas)))
        # TODO: remove the restriction of videos_per_gpu == 1 when prepared
        videos_per_gpu = kwargs['img_group_0'][0].size(0)
        assert videos_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(num_modalities, img_metas, **kwargs)
        else:
            return self.aug_test(num_modalities, img_metas, **kwargs)

    def forward(self, num_modalities, img_meta, return_loss=True, **kwargs):
        num_modalities = int(num_modalities[0])
        if return_loss:
            return self.forward_train(num_modalities, img_meta, **kwargs)
        else:
            return self.forward_test(num_modalities, img_meta, **kwargs)
