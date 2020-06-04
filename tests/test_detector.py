import os
import os.path as osp

import mmcv
import numpy as np

from mmmovie import PersonDetector


class TestDetector(object):

    @classmethod
    def setup_class(cls):
        cls.rcnn_bboxes = np.array([[363, 95, 670, 574], [922, 2, 1330, 577]],
                                   np.int16)
        cls.rcnn_conf = np.array([0.994, 0.995])
        cls.retina_bboxes = np.array(
            [[368, 96, 652, 570], [919, 0, 1329, 576]], np.int16)
        cls.retina_conf = np.array([0.879, 0.866])

    def test_rcnn(self):
        cfg = './model/cascade_rcnn_x101_64x4d_fpn.json'
        assert osp.isfile(cfg)
        weight = './model/cascade_rcnn_x101_64x4d_fpn.pth'
        assert osp.isfile(weight)
        detector = PersonDetector('rcnn', cfg, weight)
        assert detector is not None
        img_path = osp.join(osp.dirname(__file__), 'data/test.jpg')
        assert osp.isfile(img_path)
        img = mmcv.imread(img_path)
        results = detector.detect(img)
        assert results.shape[0] == 2
        bboxes = results[:, :4].astype(np.int16)
        confs = (results[:, 4] * 1000).astype(np.int16)
        assert (bboxes == self.rcnn_bboxes).all()
        assert (confs == (self.rcnn_conf * 1000).astype(np.int16)).all()

    def test_retina(self):
        cfg = osp.join(os.getcwd(), 'model/retinanet_r50_fpn.json')
        assert osp.isfile(cfg)
        weight = osp.join(os.getcwd(), 'model/retinanet_r50_fpn.pth')
        assert osp.isfile(weight)
        detector = PersonDetector('retina', cfg, weight)
        assert detector is not None
        img_path = osp.join(osp.dirname(__file__), 'data/test.jpg')
        assert osp.isfile(img_path)
        img = mmcv.imread(img_path)
        results = detector.detect(img)
        assert results.shape[0] == 2
        bboxes = results[:, :4].astype(np.int16)
        confs = (results[:, 4] * 1000).astype(np.int16)
        assert (bboxes == self.retina_bboxes).all()
        assert (confs == (self.retina_conf * 1000).astype(np.int16)).all()
