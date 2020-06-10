from __future__ import division

import torch.nn as nn

from .base import BaseDetector
from .modules.convfc_bbox_head import SharedFCBBoxHead
from .modules.core.bbox_nms import multiclass_nms
from .modules.core.bbox_transform import bbox2result, bbox2roi, bbox_mapping
from .modules.core.merge_augs import merge_aug_bboxes
from .modules.fpn import FPN
from .modules.resnext import ResNeXt
from .modules.roi_extractor import SingleRoIExtractor
from .modules.rpn_head import RPNHead
from .test_mixins import RPNTestMixin


class CascadeRCNN(BaseDetector, RPNTestMixin):

    def __init__(self,
                 num_stages,
                 backbone_cfg,
                 neck_cfg=None,
                 rpn_head_cfg=None,
                 bbox_roi_extractor_cfg=None,
                 bbox_head_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CascadeRCNN, self).__init__()

        self.num_stages = num_stages
        self.backbone = ResNeXt(**backbone_cfg)
        self.neck = FPN(**neck_cfg)
        self.rpn_head = RPNHead(**rpn_head_cfg)

        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        if not isinstance(bbox_roi_extractor_cfg, list):
            bbox_roi_extractor_cfg = [
                bbox_roi_extractor_cfg for _ in range(num_stages)
            ]
        if not isinstance(bbox_head_cfg, list):
            bbox_head_cfg = [bbox_head_cfg for _ in range(num_stages)]
        assert len(bbox_roi_extractor_cfg) == len(bbox_head_cfg) \
            == self.num_stages
        for roi_extractor_cfg, head_cfg in zip(bbox_roi_extractor_cfg,
                                               bbox_head_cfg):
            self.bbox_roi_extractor.append(
                SingleRoIExtractor(**roi_extractor_cfg))
            self.bbox_head.append(SharedFCBBoxHead(**head_cfg))

        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(CascadeRCNN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def simple_test(self, img, img_meta, proposals=None, rescale=True):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta,
            self.test_cfg['rpn']) if proposals is None else proposals

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg['rcnn']

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            bbox_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], rois)

            cls_score, bbox_pred = bbox_head(bbox_feats)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])

        cls_score = sum(ms_scores) / self.num_stages
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result
        results = ms_bbox_result['ensemble']
        return results[0]

    def aug_test(self, imgs, img_metas, proposals=None, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes will fit the scale of
        imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg['rpn'])

        rcnn_test_cfg = self.test_cfg['rcnn']
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(self.extract_feats(imgs), img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_roi_extractor = self.bbox_roi_extractor[i]
                bbox_head = self.bbox_head[i]

                bbox_feats = bbox_roi_extractor(
                    x[:len(bbox_roi_extractor.featmap_strides)], rois)
                if self.with_shared_head:
                    bbox_feats = self.shared_head(bbox_feats)

                cls_score, bbox_pred = bbox_head(bbox_feats)
                ms_scores.append(cls_score)

                if i < self.num_stages - 1:
                    bbox_label = cls_score.argmax(dim=1)
                    rois = bbox_head.regress_by_class(rois, bbox_label,
                                                      bbox_pred, img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg['score_thr'],
                                                rcnn_test_cfg['nms'],
                                                rcnn_test_cfg['max_per_img'])

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        return bbox_result[0]
