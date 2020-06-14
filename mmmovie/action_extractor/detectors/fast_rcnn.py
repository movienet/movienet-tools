import torch.nn as nn

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin
from ..core.bbox2d.transforms import bbox2result

from ..tenons.backbones import ResNet_I3D
from ..tenons.shared_heads import ResI3DLayer
from ..tenons.bbox_heads import BBoxHead
from ..tenons.roi_extractors import SingleRoIStraight3DExtractor


class FastRCNN(BaseDetector, RPNTestMixin, BBoxTestMixin):

    def __init__(self,
                 backbone,
                 bbox_roi_extractor,
                 bbox_head,
                 test_cfg,
                 shared_head=None,
                 pretrained=None):

        super(FastRCNN, self).__init__()
        self.backbone = ResNet_I3D(**backbone)
        self.shared_head = ResI3DLayer(**shared_head)
        self.bbox_roi_extractor = SingleRoIStraight3DExtractor(
            **bbox_roi_extractor)
        self.bbox_head = BBoxHead(**bbox_head)
        self.test_cfg = test_cfg

    def extract_feat(self, image_group):
        x = self.backbone(image_group)
        if self.with_neck:
            x = self.neck()
        else:
            if not isinstance(x, (list, tuple)):
                x = (x, )
        return x

    def simple_test(self,
                    img_meta,
                    proposals=None,
                    rescale=False,
                    num_modalities=1,
                    **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        img_group = kwargs['img_group_0'][0]
        x = self.extract_feat(img_group)

        proposal_list = []
        for proposal in proposals:
            proposal = proposal[0, ...]
            # if not self.test_cfg.train_detector:
            #     select_inds = proposal[:, 4] >= min(
            #         self.test_cfg.person_det_score_thr, max(proposal[:, 4]))
            #     proposal = proposal[select_inds]
            proposal_list.append(proposal)

        img_meta = img_meta[0]

        det_bboxes, det_labels, feature = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        # bbox_results = bbox2result(
        #     det_bboxes,
        #     det_labels,
        #     self.bbox_head.num_classes,
        #     thr=self.test_cfg.rcnn.action_thr)
        det_labels = det_labels[:, 1:]

        return det_bboxes, det_labels, feature

    def aug_test(self,
                 img_metas,
                 proposals=None,
                 rescale=False,
                 num_modalities=1,
                 **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes will fit the scale
        of imgs[0]
        """
        img_groups = kwargs['img_group_0']
        if proposals is None:
            proposal_list = self.aug_test_rpn(
                self.extract_feats(img_groups), img_metas, self.test_cfg.rpn)
        else:
            # TODO: need check
            proposal_list = []
            for proposal in proposals:
                proposal = proposal[0, ...]
                # if not self.test_cfg.train_detector:
                #     select_inds = proposal[:, 4] >= min(
                #         self.test_cfg.person_det_score_thr, max(
                #             proposal[:, 4]))
                #     proposal = proposal[select_inds]
                proposal_list.append(proposal)

        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(img_groups), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        # bbox_results = bbox2result(
        #     _det_bboxes,
        #     det_labels,
        #     self.bbox_head.num_classes,
        #     thr=self.test_cfg.rcnn.action_thr)

        # return bbox_results
        det_labels = det_labels[:, 1:]

        return det_bboxes, det_labels, feature
