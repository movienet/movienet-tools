from mmmovie.metaio.tracklet import ShotLevelTrackletSet
from .src.dataset import ActionDataset


class ActionExtractManager(object):

    def __init__(self):
        pass

    def run_detect(self, detector, video, shot_file):
        dataset = ActionDataset(video, tracklet_file=None, shot_file=shot_file)
        img_list = dataset.get_det_infos()
        result = detector.batch_detect(img_list, '')
        tracklets = self._bbox_result_to_tracklets(result)
        return tracklets

    def _bbox_result_to_tracklets(self, result):
        pass

    def run_extract(self, extractor, video, shot_file, tracklet_file):
        dataset = ActionDataset(
            video, tracklet_file=tracklet_file, shot_file=shot_file)
        result = extractor.extract(dataset)
        return result
