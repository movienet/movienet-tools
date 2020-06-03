from mmmovie import PersonDetector


class TestDetector(object):

    @classmethod
    def setup_class(cls):
        cls.mid = 'tt0120338'
        cls.tmdb_id = '597'
        cls.douban_id = '1292722'

    def test_person_detector(self):
        cfg = './model/cascade_rcnn_x101_64x4d_fpn.json'
        weight = './model/cascade_rcnn_x101_64x4d_fpn.pth'
        detector = PersonDetector('rcnn', cfg, weight)
        assert detector is not None
