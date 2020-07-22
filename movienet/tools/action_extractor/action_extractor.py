import mmcv
from .detectors import FastRCNN
import torch
from .src import ActionDataPreprocessor
import numpy as np
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader
from functools import partial
from itertools import groupby
from operator import itemgetter


class ActionExtractor(object):

    def __init__(self, config_path, weight_path, gpu=0):

        cfg = mmcv.Config.fromfile(config_path)
        self.cfg = cfg
        self.gpu = gpu
        self.model = FastRCNN(**cfg)
        self.model.eval()
        self.model.cuda(gpu)
        mmcv.runner.load_checkpoint(
            self.model, weight_path, map_location=f"cuda:{gpu}")
        self.data_preprocessor = ActionDataPreprocessor(gpu)

    def extract(self, imgs, bboxes):
        """
        imgs: sequence of image
        bboxes: static tracklet, expressed by boxes within one frame.
        """
        assert np.logical_and(bboxes <= 1, bboxes >= 0).all()
        data = self.data_preprocessor(imgs, bboxes)
        with torch.no_grad():
            scaled_bboxes, score, feature = self.model(rescale=True, **data)
        assert len(scaled_bboxes) == len(score) == len(feature)
        return [
            dict(
                bboxes=_box.cpu().numpy(),
                score=_score.cpu().numpy(),
                action_feature=_feat.cpu().numpy())
            for _box, _score, _feat in zip(scaled_bboxes, score, feature)
        ]


class ParallelActionExtractor(object):

    def __init__(self, config_path, weight_path, gpu_ids=None):

        cfg = mmcv.Config.fromfile(config_path)
        self.cfg = cfg
        if gpu_ids is None:
            self.gpu_ids = list(range(torch.cuda.device_count()))
        else:
            self.gpu_ids = gpu_ids
        assert isinstance(self.gpu_ids, list)
        self.ngpu = len(self.gpu_ids)
        self.model = FastRCNN(**cfg)
        mmcv.runner.load_checkpoint(
            self.model, weight_path, map_location="cpu")
        self.model = MMDataParallel(
            self.model.cuda(self.gpu_ids[0]), device_ids=self.gpu_ids)
        self.model.eval()

    def extract(self, dataset, workers_per_gpu=4):
        data_loader = DataLoader(
            dataset,
            batch_size=self.ngpu,
            sampler=None,
            num_workers=workers_per_gpu * self.ngpu,
            collate_fn=partial(collate, samples_per_gpu=1),
            pin_memory=False,
            worker_init_fn=None)
        results = []
        prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                scaled_bboxes, score, feature = self.model(
                    rescale=True, **data)
            results.append(
                dict(
                    bboxes=scaled_bboxes.cpu().numpy(),
                    score=score.cpu().numpy(),
                    action_feature=feature.cpu().numpy()))
            for i in range(self.ngpu):
                prog_bar.update()
        bbox_tracklet_ids = data_loader.dataset.bbox_tracklet_ids
        shot_group_slice = data_loader.dataset.shot_group_slice
        ret = self._post_process(results, bbox_tracklet_ids, shot_group_slice)
        return ret

    def _post_process(self, results, ids, groups):

        def _merge_result(_bbox, _score, _feat, _id):
            ret = []
            for key, grp in groupby(enumerate(_id), key=itemgetter(1)):
                grp = [g[0] for g in list(grp)]
                tracklet = _bbox[grp]
                avg_score = _score[grp].mean(axis=0)
                max_score = _score[grp].max(axis=0)
                feat = _feat[grp].mean(axis=0)
                ret.append(
                    dict(
                        tracklet=tracklet,
                        tracklet_id=key,
                        avg_score=avg_score,
                        max_score=max_score,
                        feat=feat))
            return ret

        bboxes, scores, feats = [], [], []
        for rst in results:
            bboxes.append(rst['bboxes'])
            scores.append(rst['score'])
            feats.append(rst['action_feature'])

        shot_level_result = []
        for group in groups:
            st, ed = group
            if st == ed:
                shot_level_result.append(None)
                continue
            this_bboxes = np.concatenate(bboxes[st:ed], axis=0)
            this_scores = np.concatenate(scores[st:ed], axis=0)
            this_feats = np.concatenate(feats[st:ed], axis=0)
            this_ids = np.concatenate(ids[st:ed], axis=0)
            shot_result_lst = _merge_result(this_bboxes, this_scores,
                                            this_feats, this_ids)
            shot_level_result.append(shot_result_lst)
        return shot_level_result
