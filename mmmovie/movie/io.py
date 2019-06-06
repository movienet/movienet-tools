import os.path as osp
from collections import OrderedDict

import cv2

from mmcv.utils import (scandir, check_file_exist, mkdir_or_exist,
                        track_progress)
from mmcv.opencv_info import USE_OPENCV2

if not USE_OPENCV2:
    from cv2 import (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS,
                     CAP_PROP_FRAME_COUNT, CAP_PROP_FOURCC,
                     CAP_PROP_POS_FRAMES, VideoWriter_fourcc)
else:
    from cv2.cv import CV_CAP_PROP_FRAME_WIDTH as CAP_PROP_FRAME_WIDTH
    from cv2.cv import CV_CAP_PROP_FRAME_HEIGHT as CAP_PROP_FRAME_HEIGHT
    from cv2.cv import CV_CAP_PROP_FPS as CAP_PROP_FPS
    from cv2.cv import CV_CAP_PROP_FRAME_COUNT as CAP_PROP_FRAME_COUNT
    from cv2.cv import CV_CAP_PROP_FOURCC as CAP_PROP_FOURCC
    from cv2.cv import CV_CAP_PROP_POS_FRAMES as CAP_PROP_POS_FRAMES
    from cv2.cv import CV_FOURCC as VideoWriter_fourcc

from mmcv.video import VideoReader


class MovieReader(VideoReader):
    def __init__(self, filename, cache_capacity=10):
        super(MovieReader, self).__init__(filename, cache_capacity)
    
    def _adjust_size(self, size):
        w, h = size
        assert isinstance(h) and (h == -1 or h > 0)
        assert isinstance(w) and (w == -1 or w > 0)
        assert not (h == -1 and w == -1)

        if h > 0 and w > 0:
            return size

        ori_h = self.height
        ori_w = self.width

        if ori_h == -1:
            scale = w / ori_w
            h = int(ori_h * scale + 0.5)
        else:
            scale = h / ori_h
            w  = int(ori_w * scale)
        return (w, h)

    
    def cvt2frames(self,
                   frame_dir,
                   size=None,
                   step=1,
                   start=0,
                   end=None,
                   filename_tmpl='{:06d}.jpg'):
        """ This function overwrites original ``mmcv.VideoReader.cvt2frame``.
        This function converts a video to frames and save the frames in a 
        perticular pattern.

        Args:
            frame_dir (str): Output root directory to store the frames.
            size (None or tuple <width, height>): frame size, if it is not None, 
                the frame will be resized. if height is -1 or width is -1,
                the ratio will be kept.
            step (int): frequency of saving frames.
            start (int): start index.
            end (int or None): end index.
            filename_tmpl (str or callable): filename template. It should be
                a formated string or some callable function/object that 
                accept an index as input parameter.
        """
        assert isinstance(start, int) and start >= 0
        assert end is None or isinstance(end, int)
        assert isinstance(step, int) and step >= 1
        _callable_fn_tmpl = callable(filename_tmpl)

        if size is not None:
            size = self._adjust_size(size)
        mkdir_or_exist(frame_dir)

        end = min(end, self.frame_cnt)
        for i in range(start, end, step):
            img = self.get_frame(i)
            if size:
                img = cv2.resize(img, size)
            if _callable_fn_tmpl:
                fn = filename_tmpl(i)
            else:
                fn = filename_tmpl.format(i)
            cv2.imwrite(osp.join(frame_dir, fn), img)
            



