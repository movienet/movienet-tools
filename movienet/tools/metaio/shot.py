import mmcv


def num_to_interval(interval, n):
    """ Interval: start index of each interval
    """
    main_id = sum([n - x >= 0 for x in interval]) - 1
    sub_id = n - interval[main_id]
    return main_id, sub_id


def parse_shot(path):
    data = mmcv.list_from_file(path)
    shots = [list(map(int, s.split())) for s in data]
    return shots


class Shot(object):

    def __init__(self, index, shot_tuple, fps=None):
        self.shot_tuple = shot_tuple
        self.fps = fps
        self.shot_index = index

    @property
    def start_frame(self):
        return self.shot_tuple[0]

    @property
    def end_frame(self):
        return self.shot_tuple[1]

    @property
    def keyframes(self):
        return self.shot_tuple[2:5]

    @property
    def nframe(self):
        return self.end_frame - self.start_frame + 1

    @property
    def start_time(self):
        assert self.fps is not None, "FPS unkown, cannot get start time."
        return self.shot_tuple[0] / self.fps

    @property
    def end_time(self):
        assert self.fps is not None, "FPS unkown, cannot get end time."
        return self.shot_tuple[1] / self.fps

    @property
    def index(self):
        """ global index
        """
        return self.shot_index


class ShotList(object):

    @staticmethod
    def from_file(path, fps=None):
        shot_numbers = parse_shot(path)
        return ShotList(shot_numbers, fps)

    def __init__(self, shot_list, fps=None):
        self.shot_list = shot_list
        self.fps = fps
        self.shots = [Shot(idx, s, fps) for idx, s in enumerate(shot_list)]

    def __iter__(self):
        for s in self.shots:
            yield s

    def __getitem__(self, idx):
        return self.shots[idx]

    def __len__(self):
        return len(self.shots)

    def frame_idx_to_shot_idx(self, frame_idx):
        starts = [s.start_frame for s in self.shots]
        shot_id, frame_id = num_to_interval(starts, frame_idx)
        return shot_id, frame_id
