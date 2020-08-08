import torch
import numpy as np
import mmcv
from collections import Sequence
from ..core.bbox2d.transforms import bbox_flip


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


class NormBBox(object):
    """ Normalize bounding box to [0, 1).
    """

    def __init__(self, input_normed=True):
        self.input_normed = input_normed

    def __call__(self, results):
        bboxes = results['bboxes']
        if self.input_normed:
            assert np.logical_and(bboxes <= 1, bboxes >= 0).all()
            return results
        height, width, _ = results['ori_shape']
        if bboxes.shape[1] == 5:
            bboxes = bboxes / np.array([width, height, width, height, 1.0],
                                       dtype=np.float32)
        else:
            bboxes = bboxes / np.array([width, height, width, height],
                                       dtype=np.float32)
        results['bboxes'] = bboxes

        return results


class BboxTransform(object):
    """Preprocess gt bboxes.
    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def _preprocess_bboxes(self, proposal, height, width):
        proposal = proposal.astype(np.float32)
        if proposal.shape[1] == 5:
            proposal = proposal * np.array([width, height, width, height, 1.0],
                                           dtype=np.float32)
            score = proposal[:, 4, None]
            proposal = proposal[:, :4]
        else:
            proposal = proposal * np.array([width, height, width, height],
                                           dtype=np.float32)
            score = None
        return proposal, score

    def __call__(self, results):
        height, width, _ = results["ori_shape"]
        bboxes, score = self._preprocess_bboxes(results["bboxes"], height,
                                                width)
        gt_bboxes = bboxes * results["scale_factor"]

        img_shape = results['img_shape']
        if results["flip"]:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape)
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)
        if self.max_num_gts is None:
            rst = gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            rst = padded_bboxes
        proposal = np.hstack([rst, score]) if score is not None else rst

        results["proposals"] = [to_tensor(proposal)]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += "(max_num_gts={})".format(self.max_num_gts)


class Images2FixedLengthGroup(object):
    """ Sample raw image list to image group.

    Sample a group of images (length = ``group_length``) from a list of image 
    (arbitary number). Images are sampled from the center frame of the image
    sequence and according to the stride ``step``.

    Args:
        TBD
    """

    def __init__(self, group_length=32, step=2, skip_offset=0):
        self.scope_length = group_length * step
        self.step = step
        self.skip_offset = skip_offset

    def _get_sample_index(self, indice, nimg):
        sampled_idxes = list()
        #
        p = max(0, indice - self.step)
        for i, ind in enumerate(
                range(-2, -(self.scope_length + 1) // 2, -self.step)):
            sampled_idxes = [p + self.skip_offset] + sampled_idxes
            if p - self.step >= 0:
                p -= self.step
        p = min(indice, nimg - 1)
        for i, ind in enumerate(
                range(0, (self.scope_length + 1) // 2, self.step)):
            sampled_idxes.append(p + self.skip_offset)
            if p + self.step < nimg:
                p += self.step
        return sampled_idxes

    def __call__(self, results):
        nimg = results['nimg']
        # indice = (nimg - 1) // 2
        indice = (nimg) // 2

        sampled_idxes = self._get_sample_index(indice, nimg)
        imgs = results['imgs']
        img_group = [imgs[p] for p in sampled_idxes]
        results['img_group'] = img_group  # [to_tensor(img_group)]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += "(scope_length={}, step={}, skip_offset={})".format(
            self.scope_length, self.step, self.skip_offset)


class LoadImages(object):

    def __init__(self, record_ori_shape=True):
        self.record_ori_shape = record_ori_shape

    def __call__(self, results):

        img_group = results['img_group']
        if isinstance(img_group[0], str):
            img_group = [mmcv.imread(img_fn) for img_fn in img_group]
        # else is np.array, image already loaded, do nothing.
        results['img_group'] = img_group
        if self.record_ori_shape:
            results['ori_shape'] = img_group[0].shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(record_ori_shape={self.record_ori_shape})"


class ImageGroupTransform(object):
    """Preprocess a group of images.
    1. rescale the images to expected size
    2. (for classification networks) crop the images with a given size
    3. flip the images (if needed)
    4(a) divided by 255 (0-255 => 0-1, if needed)
    4. normalize the images
    5. pad the images (if needed)
    6. transpose to (c, h, w)
    7. stack to (N, c, h, w)
    where, N = 1 * N_oversample * N_seg * L
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None,
                 scale=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor
        self.scale = scale
        # self.resize_crop = resize_crop
        # self.rescale_crop = rescale_crop

        # croping parameters
        # if crop_size is not None:
        #     if oversample == 'three_crop':
        #         self.op_crop = Group3CropSample(crop_size)
        #     elif oversample == 'ten_crop':
        #         # oversample crop (test)
        #         self.op_crop = GroupOverSample(crop_size)
        #     elif resize_crop:
        #         self.op_crop = RandomResizedCrop(crop_size)
        #     elif rescale_crop:
        #         self.op_crop = RandomRescaledCrop(crop_size)
        #     elif multiscale_crop:
        #         # multiscale crop (train)
        #         self.op_crop = GroupMultiScaleCrop(
        #             crop_size, scales=scales, max_distort=max_distort,
        #             fix_crop=not random_crop, more_fix_crop=more_fix_crop)
        #     else:
        #         # center crop (val)
        #         self.op_crop = GroupCenterCrop(crop_size)
        # else:
        # self.op_crop = None

    def __call__(self, results):

        # if self.resize_crop or self.rescale_crop:
        #     img_group, crop_quadruple = self.op_crop(img_group)
        #     img_shape = img_group[0].shape
        #     scale_factor = None
        # else:
        # 1. rescale
        # if keep_ratio:
        img_group = results['img_group']
        tuple_list = [
            mmcv.imrescale(img, self.scale, return_scale=True)
            for img in img_group
        ]
        img_group, scale_factors = list(zip(*tuple_list))
        scale_factor = scale_factors[0]
        # else:
        #     tuple_list = [mmcv.imresize(
        #         img, scale, return_scale=True) for img in img_group]
        #     img_group, w_scales, h_scales = list(zip(*tuple_list))
        #     scale_factor = np.array([w_scales[0], h_scales[0],
        #                                 w_scales[0], h_scales[0]],
        #                             dtype=np.float32)
        # 2. crop (if necessary)
        # if crop_history is not None:
        #     self.op_crop = GroupCrop(crop_history)
        # if self.op_crop is not None:
        #     img_group, crop_quadruple = self.op_crop(
        #         img_group, is_flow=is_flow)
        # else:
        crop_quadruple = None
        img_shape = img_group[0].shape
        # # 3. flip
        # if flip:
        #     img_group = [mmcv.imflip(img) for img in img_group]
        # if is_flow:
        #     for i in range(0, len(img_group), 2):
        #         img_group[i] = mmcv.iminvert(img_group[i])
        # # 4a. div_255
        # if div_255:
        #     img_group = [mmcv.imnormalize(img, 0, 255, False)
        #                  for img in img_group]
        # 4. normalize
        img_group = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in img_group
        ]
        # 5. pad
        if self.size_divisor is not None:
            img_group = [
                mmcv.impad_to_multiple(img, self.size_divisor)
                for img in img_group
            ]
            pad_shape = img_group[0].shape
        else:
            pad_shape = img_shape
        # if is_flow:
        #     assert len(img_group[0].shape) == 2
        #     img_group = [np.stack((flow_x, flow_y), axis=2)
        #                  for flow_x, flow_y in zip(
        #                      img_group[0::2], img_group[1::2])]
        # 6. transpose
        img_group = [img.transpose(2, 0, 1) for img in img_group]

        # Stack into numpy.array
        img_group = np.stack(img_group, axis=0)
        img_group = to_tensor(img_group)
        img_group = np.transpose(img_group, (1, 0, 2, 3))

        results['img_group_0'] = [img_group]  #img_group
        results['img_shape'] = img_shape
        results['pad_shape'] = pad_shape
        results['scale_factor'] = scale_factor
        results['crop_quadruple'] = crop_quadruple
        results['flip'] = False
        return results
