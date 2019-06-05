import os
import os.path as osp
import tempfile

import sys
sys.path.append('..')
import mmmovie
from mmmovie import MovieReader
import pytest
import math


class TestMovie(object):

    @classmethod
    def setup_class(cls):
        cls.video_path = osp.join(osp.dirname(__file__), 'data/test.mp4')
        cls.num_frames = 482

    def test_load(self):
        v = MovieReader(self.video_path)
        print('width: {}, height: {}, fps: {}, frame_cnt: {}'.format(
            v.width, v.height, v.fps, v.frame_cnt))
        assert v.width == 426
        assert v.height == 240
        assert round(v.fps) == 24
        assert v.frame_cnt == self.num_frames
        assert len(v) == self.num_frames
        assert v.opened
        import cv2
        assert isinstance(v.vcap, type(cv2.VideoCapture()))
    
    def test_read(self):
        v = MovieReader(self.video_path)
        img = v.read()
        assert int(round(img.mean())) == 41
        img = v.get_frame(127)
        assert int(round(img.mean())) == 43
        img = v[128]
        assert int(round(img.mean())) == 42
        img = v[-354]
        assert int(round(img.mean())) == 42
        img = v[-355]
        assert int(round(img.mean())) == 43
        img = v.read()
        assert int(round(img.mean())) == 42
        with pytest.raises(IndexError):
            v.get_frame(self.num_frames + 1)
        with pytest.raises(IndexError):
            v[-self.num_frames - 1]
    
    def test_slice(self):
        v = MovieReader(self.video_path)
        imgs = v[-355:-353]
        assert int(round(imgs[0].mean())) == 43
        assert int(round(imgs[1].mean())) == 42
        assert len(imgs) == 2
        imgs = v[127:129]
        assert int(round(imgs[0].mean())) == 43
        assert int(round(imgs[1].mean())) == 42
        assert len(imgs) == 2
        imgs = v[128:126:-1]
        assert int(round(imgs[0].mean())) == 42
        assert int(round(imgs[1].mean())) == 43
        assert len(imgs) == 2
        imgs = v[:5]
        assert len(imgs) == 5
        imgs = v[480:]
        assert len(imgs) == 2
        imgs = v[-3:]
        assert len(imgs) == 3

    def test_current_frame(self):
        v = MovieReader(self.video_path)
        assert v.current_frame() is None
        v.read()
        img = v.current_frame()
        assert int(round(img.mean())) == 41

    def test_position(self):
        v = MovieReader(self.video_path)
        assert v.position == 0
        for _ in range(10):
            v.read()
        assert v.position == 10
        v.get_frame(99)
        assert v.position == 100
    
    def test_iterator(self):
        cnt = 0
        for img in MovieReader(self.video_path):
            cnt += 1
            assert img.shape == (240, 426, 3)
        assert cnt == self.num_frames

    def test_with(self):
        with MovieReader(self.video_path) as v:
            assert v.opened
        assert not v.opened
    
    def test_cvt2frames(self):
        v = MovieReader(self.video_path)
        frame_dir = tempfile.mkdtemp()
        v.cvt2frames(frame_dir)
        assert osp.isdir(frame_dir)
        for i in range(self.num_frames):
            filename = '{}/{:06d}.jpg'.format(frame_dir, i)
            assert osp.isfile(filename)
            os.remove(filename)

        v = MovieReader(self.video_path)
        v.cvt2frames(frame_dir, show_progress=False)
        assert osp.isdir(frame_dir)
        for i in range(self.num_frames):
            filename = '{}/{:06d}.jpg'.format(frame_dir, i)
            assert osp.isfile(filename)
            os.remove(filename)

        v = MovieReader(self.video_path)
        v.cvt2frames(
            frame_dir,
            file_start=100,
            filename_tmpl='{:03d}.JPEG',
            start=100,
            max_num=20)
        assert osp.isdir(frame_dir)
        for i in range(100, 120):
            filename = '{}/{:03d}.JPEG'.format(frame_dir, i)
            assert osp.isfile(filename)
            os.remove(filename)
        os.removedirs(frame_dir)
    
    def test_resize_video(self):
        out_file = osp.join(tempfile.gettempdir(), '.mmmovie_test.mp4')
        mmmovie.resize_movie(self.video_path, out_file, (480, 360), log_level='error', quiet=True)
        v = MovieReader(out_file)
        assert v.resolution == (480, 360)
        os.remove(out_file)
        mmmovie.resize_movie(self.video_path, out_file, ratio=2, log_level='error')
        v = MovieReader(out_file)
        assert v.resolution == (852, 480)
        os.remove(out_file)
        mmmovie.resize_movie(self.video_path, out_file, (1000, 480), keep_ar=True, log_level='error')
        v = MovieReader(out_file)
        assert v.resolution == (852, 480)
        os.remove(out_file)
        mmmovie.resize_movie(
            self.video_path, out_file, ratio=(3, 2), keep_ar=True, log_level='error')
        v = MovieReader(out_file)
        assert v.resolution == (1278, 480)
        os.remove(out_file)
        mmmovie.resize_movie(
            self.video_path, out_file, size='240P', log_level='error')
        v = MovieReader(out_file)
        print(v.resolution)
        assert v.resolution == (352, 240)
        os.remove(out_file)
        mmmovie.resize_movie(
            self.video_path, out_file, size='240P',  keep_ar=True, log_level='error')
        v = MovieReader(out_file)
        assert v.resolution == (426, 240)
        os.remove(out_file)
