import mmcv
import decord
import os
from movienet.tools.metaio import ShotList


class SimpleFileGenerator(object):

    def __init__(self, data_root, tmpl, **kwargs):
        self.data_root = data_root
        self.tmpl = tmpl

    def __call__(self, idx):
        return os.path.join(self.data_root, self.tmpl.format(idx))


class TwoLevelFileGenerator(object):

    def __init__(self, data_root, tmpl, shot_file):
        self.data_root = data_root
        self.tmpl = tmpl
        self.shot_list = ShotList.from_file(shot_file)

    def __call__(self, idx):
        sid, fid = self.shot_list.frame_idx_to_shot_idx(idx)
        return os.path.join(self.data_root, self.tmpl.format(sid, fid))


class VideoFileBackend(object):

    def __init__(self,
                 file_generator_type,
                 data_root,
                 tmpl='shot_{:04d}/{:06d}.jpg',
                 shot_file=None):
        if file_generator_type == 'simple':
            self.file_generator = SimpleFileGenerator(data_root, tmpl)
        elif file_generator_type == 'twolevel':
            self.file_generator = TwoLevelFileGenerator(
                data_root, tmpl, shot_file)

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            return [
                self.file_generator(i)
                for i in range(subscript.start, subscript.stop, subscript.step)
            ]
        else:
            return self.file_generator(subscript)


class VideoMMCVBackend(object):

    def __init__(self, video_path):
        self.video_path = video_path
        self.video = mmcv.VideoReader(video_path)

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            return [
                self.video[i]
                for i in range(subscript.start, subscript.stop, subscript.step)
            ]
        else:
            return self.video[subscript]

    def to_config(self):
        return mmcv.Config(
            dict(
                type='VideoMMCVBackend',
                params=dict(video_path=self.video_path)))


class VideoDecordBackend(object):

    def __init__(self, video_path):
        self.video_path = video_path
        self.video = decord.VideoReader(video_path)

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            return [
                self.video[i]
                for i in range(subscript.start, subscript.stop, subscript.step)
            ]
        else:
            return self.video[subscript]

    def to_config(self):
        return mmcv.Config(
            dict(
                type='VideoDecordBackend',
                params=dict(video_path=self.video_path)))