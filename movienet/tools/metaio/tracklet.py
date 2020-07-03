from movienet.tools.utils import bbox_overlaps, bimatch


class ShotLevelTrackletSet(object):

    def __init__(self, bbox_lst, frame_id_lst, id_prefix=None, thr=0.7):
        self.bboxes = bbox_lst
        self.frame_ids = frame_id_lst
        self.id_prefix = id_prefix
        self.thr = thr  # tracklet overlap threshold
        self._clean_bboxes()
        self.set = self._match()
        (self.frame_based_indexing,
         self.id_based_indexing) = self._build_indexing()

    def _clean_bboxes(self):
        assert len(self.bboxes) == len(self.frame_ids)
        tmp_bboxes, tmp_ids = [], []
        for _bbox, _frame_id in zip(self.bboxes, self.frame_ids):
            if len(_bbox) == 0:
                continue
            tmp_bboxes.append(_bbox)
            tmp_ids.append(_frame_id)
        self.bboxes = tmp_bboxes
        self.frame_ids = tmp_ids

    def _match(self):
        if len(self.bboxes) == 0:
            return []

        tset = []  # tracklet set
        id_strings = [
            f"{self.id_prefix}_{j}" for j in range(len(self.bboxes[0]))
        ]
        tset.append((self.frame_ids[0], self.bboxes[0], id_strings))
        tidx = len(self.bboxes[0])  # tracklet global id
        for i in range(1, len(self.bboxes)):

            last_bboxes = self.bboxes[i - 1]
            last_ids = tset[i - 1][2]
            this_bboxes = self.bboxes[i]
            linked_idx, new_idx = self._iou_bimatch(last_bboxes, this_bboxes)
            this_tids = [None] * len(this_bboxes)
            for this_idx, matched_idx in linked_idx:
                this_tids[this_idx] = last_ids[matched_idx]
            for this_idx in new_idx:
                this_tids[this_idx] = f"{self.id_prefix}_{tidx}"
                tidx += 1
            tset.append((self.frame_ids[i], this_bboxes, this_tids))
        return tset

    def _iou_bimatch(self, box1, box2):
        ious = bbox_overlaps(box2, box1)
        match_result, _ = bimatch(ious, self.thr)
        linked_idx, new_idx = [], []
        for i, mi in enumerate(match_result):
            if mi == -1:
                new_idx.append(i)
            else:
                linked_idx.append((i, mi))
        return linked_idx, new_idx

    def _build_indexing(self):
        frame_based_indexing = {
            frame_id: (bboxes, tids)
            for frame_id, bboxes, tids in self.set
        }
        id_based_indexing = None  # not implemented
        return frame_based_indexing, id_based_indexing

    def get_bboxes(self, center_lst):
        ret = []
        for c in center_lst:
            box = self.frame_based_indexing.get(c, None)
            if box is None:
                ret.append(None)
            else:
                ret.append(box[0])
        return ret

    def get_tids(self, center_lst):
        ret = []
        for c in center_lst:
            box = self.frame_based_indexing.get(c, None)
            if box is None:
                ret.append(None)
            else:
                ret.append(box[1])
        return ret
