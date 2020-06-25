from mmmovie.metaio.tracklet import ShotLevelTrackletSet
from .src.dataset import ActionDataset


class ActionExtractManager(object):

    def __init__(self):
        pass

    def run_detect(self, detector, video, shot_file, imgs_per_gpu=1):
        dataset = ActionDataset(video, tracklet_file=None, shot_file=shot_file)
        img_list = dataset.get_det_infos()
        result = detector.batch_detect(img_list, '', imgs_per_gpu=imgs_per_gpu)
        from IPython import embed
        embed()
        tracklets = self._bbox_result_to_tracklets(result,
                                                   dataset.shot_group_slice,
                                                   dataset.sequence_centers)
        return tracklets

    def _bbox_result_to_tracklets(self, result, group, frame_ids):
        tracklets = []
        for i, (st, ed) in enumerate(group):
            this_rst = result[st:ed]
            this_frame_ids = frame_ids[st:ed]
            tracklet_set = ShotLevelTrackletSet(
                this_rst, this_frame_ids, id_prefix=f"shot_{i}")
            tracklets.append(tracklet_set)
        return tracklets

    def run_extract(self, extractor, video, shot_file, tracklet_file):
        dataset = ActionDataset(
            video, tracklet_file=tracklet_file, shot_file=shot_file)
        result = extractor.extract(dataset)
        return result
