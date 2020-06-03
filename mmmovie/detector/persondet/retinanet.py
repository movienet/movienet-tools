import torch.nn as nn

from .base import BaseDetector
from .modules.core.bbox_transform import bbox2result
from .modules.fpn import FPN
from .modules.resnet import ResNet
from .modules.retina_head import RetinaHead


class RetinaNet(BaseDetector):

    def __init__(self,
                 backbone_cfg,
                 neck_cfg,
                 head_cfg,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet, self).__init__()
        self.backbone = ResNet(**backbone_cfg)
        self.neck = FPN(**neck_cfg)
        self.bbox_head = RetinaHead(**head_cfg)
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(RetinaNet, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
