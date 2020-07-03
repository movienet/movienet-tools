import os
import os.path as osp

from movienet.tools import ShotDetector


class TestShotDetector(object):

    @classmethod
    def setup_class(cls):
        cls.video_path = osp.join(osp.dirname(__file__), 'data/test1.mp4')
        cls.out_dir = osp.join(osp.dirname(__file__), 'data')

    def test_shotdetect(self, ):
        sdt = ShotDetector(
            print_list=True,
            save_keyf=False,
            save_keyf_txt=True,
            split_video=False,
            begin_frame=0,
            end_frame=2000)
        sdt.shotdetect(self.video_path, self.out_dir)
        assert len(
            os.listdir(
                osp.join(osp.dirname(__file__),
                         'data/shot_keyf/test1'))) == 3 * 3
        with open(
                osp.join(osp.dirname(__file__), 'data/shot_txt/test1.txt'),
                'r') as f:
            keyf_txt_list = f.read().splitlines()
        assert len(keyf_txt_list) == 3
