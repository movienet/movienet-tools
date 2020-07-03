import mmcv
from torchvision.transforms import Compose
from .transforms import (Images2FixedLengthGroup, ImageGroupTransform,
                         BboxTransform, LoadImages)
from .formating import Collect, OneSampleCollate
from movienet.tools.metaio import ShotList, ShotLevelTrackletSet
import math


class ActionDataPreprocessor(object):

    def __init__(self, gpu=0):
        self.pipeline = self.build_data_pipline(gpu)

    def __call__(self, imgs, bboxes):
        results = dict(
            imgs=imgs,
            bboxes=bboxes,
            nimg=len(imgs),
            ori_shape=(imgs[0].shape[0], imgs[0].shape[1], 3))
        # results = self.pre_pipeline(results)
        return self.pipeline(results)

    def build_data_pipline(self, gpu):
        pipeline = Compose([
            Images2FixedLengthGroup(32, 2, 0),
            ImageGroupTransform(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
                size_divisor=32,
                scale=(800, 256)),
            BboxTransform(),
            Collect(
                keys=["img_group_0", "proposals"],
                meta_keys=[
                    "ori_shape", "img_shape", "pad_shape", "scale_factor",
                    "crop_quadruple", "flip"
                ],
                list_meta=True),
            OneSampleCollate(gpu)
        ])
        return pipeline


class ActionDataset(object):

    def __init__(self, video, tracklet_file=None, shot_file=None, seq_len=64):
        """ 
        video: video frame pool, backend could be ``file``, ``mmcv`` or
        ``decord``
        """
        self.video = video
        if shot_file is not None:
            self.shot_list = ShotList.from_file(shot_file)
        if tracklet_file is not None:
            self.tracklet_list = mmcv.load(tracklet_file)
        else:
            self.tracklet_list = None
        self.seq_len = seq_len

        (self.bbox_stream, self.bbox_tracklet_ids, self.shot_group_slice,
         self.sequence_centers, self.seq_stream) = self._init_stream()
        self.pipeline = self.build_data_pipline()

    def __len__(self):
        return len(self.seq_stream)

    def build_data_pipline(self):
        pipeline = Compose([
            Images2FixedLengthGroup(32, 2, 0),
            LoadImages(record_ori_shape=True),
            ImageGroupTransform(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
                size_divisor=32,
                scale=(800, 256)),
            BboxTransform(),
            Collect(
                keys=["img_group_0", "proposals"],
                meta_keys=[
                    "ori_shape", "img_shape", "pad_shape", "scale_factor",
                    "crop_quadruple", "flip"
                ],
                list_meta=True),
        ])
        return pipeline

    def _init_stream(self):

        def _allocate(_shot):
            _nframe = _shot.nframe
            if _nframe <= self.seq_len:
                return [[_shot.start_frame, _shot.end_frame + 1]
                        ], [_shot.start_frame + _nframe // 2]
            else:
                ret_range, ret_center = [], []
                ngroup = math.ceil(_nframe / self.seq_len)
                overlap = math.floor(
                    (self.seq_len * ngroup - _nframe) / (ngroup - 1))
                # from IPython import embed
                # embed()
                for j, i in enumerate(
                        range(_shot.start_frame, _shot.start_frame + _nframe,
                              self.seq_len)):
                    _st = max(_shot.start_frame, i - j * overlap)
                    _ed = min(_st + self.seq_len, _shot.end_frame)
                    _st = _ed - self.seq_len
                    ret_range.append([_st, _ed])
                    ret_center.append(_st + self.seq_len // 2)
                return ret_range, ret_center

        (bbox_stream, bbox_tracklet_ids, shot_group_slice, sequence_centers,
         seq_stream) = [], [], [], [], []

        for shot_id, shot in enumerate(self.shot_list):
            range_lst, center_lst = _allocate(shot)
            if self.tracklet_list:
                bboxes = self.tracklet_list[shot.index].get_bboxes(center_lst)

                tracklet_ids = self.tracklet_list[shot.index].get_tids(
                    center_lst)
                center_lst = [
                    c for c, b in zip(center_lst, bboxes) if b is not None
                ]
                range_lst = [
                    r for r, b in zip(range_lst, bboxes) if b is not None
                ]
                tracklet_ids = [t for t in tracklet_ids if t is not None]
                bboxes = [b for b in bboxes if b is not None]
                bbox_tracklet_ids.extend(tracklet_ids)
                bbox_stream.extend(bboxes)
                if shot_group_slice == []:
                    shot_group_slice.append([0, len(center_lst)])
                else:
                    last = shot_group_slice[-1]
                    shot_group_slice.append(
                        [last[1], last[1] + len(center_lst)])
                seq_stream.extend(range_lst)
            else:
                sequence_centers.extend(center_lst)
                if shot_group_slice == []:
                    shot_group_slice.append([0, len(center_lst)])
                else:
                    last = shot_group_slice[-1]
                    shot_group_slice.append(
                        [last[1], last[1] + len(center_lst)])
                seq_stream.extend(range_lst)
        return (bbox_stream, bbox_tracklet_ids, shot_group_slice,
                sequence_centers, seq_stream)

    def __getitem__(self, idx):
        imgs = self.seq_stream[idx]
        imgs = [self.video[i] for i in range(imgs[0], imgs[1])]
        results = dict(imgs=imgs, bboxes=self.bbox_stream[idx], nimg=len(imgs))
        return self.pipeline(results)

    def get_det_infos(self):
        return [self.video[i] for i in self.sequence_centers]