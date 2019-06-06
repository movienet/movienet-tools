import os
import os.path as osp

import sys
sys.path.append('..')
from mmmovie import ShotDetector
import pytest

class TestMovie(object):

    @classmethod
    def setup_class(cls):
        cls.video_path = osp.join(osp.dirname(__file__), 'data/test.mp4')
        cls.out_dir =  osp.join(osp.dirname(__file__), 'data')
    def test_shotdetect(self):
        sdt = ShotDetector(print_list=True, save_keyf=True, save_keyf_txt=True,
                            begin_frame=0,end_frame=2000)    
        sdt.shotdetect(self.video_path,self.out_dir)
        assert len(os.listdir(osp.join(osp.dirname(__file__), 'data/shot_keyf/test'))) == 3*3
        with open(osp.join(osp.dirname(__file__), 'data/shot_movie/test.txt'),"r") as f:
            keyf_txt_list = f.read().splitlines()
        assert len(keyf_txt_list) == 3