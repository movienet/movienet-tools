import mmcv
from torchvision.transforms import Compose
from .transforms import (Images2FixedLengthGroup, ImageGroupTransform,
                         BboxTransform)
from .formating import Collect, OneSampleCollate


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

    # def pre_pipeline(self, results):
    #     result = 0

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


# def _get_frames(indice, nimg):
#     # images = list()
#     sampled_idxes = list()
#     #
#     p = indice - self.new_step
#     for i, ind in enumerate(
#             range(-2, -(self.old_length + 1) // 2, -self.new_step)):
#         # seg_imgs = self._load_image(osp.join(
#         #     self.img_prefix, record['video_id']),
#         #     image_tmpl, modality, p + skip_offsets[i])
#         # images = seg_imgs + images
#         sampled_idxes = [p] + sampled_idxes
#         if p - self.new_step >= 0:
#             p -= self.new_step
#     p = indice
#     for i, ind in enumerate(
#             range(0, (self.old_length + 1) // 2, self.new_step)):
#         # seg_imgs = self._load_image(osp.join(
#         #     self.img_prefix, record['video_id']),
#         #     image_tmpl, modality, p + skip_offsets[i])
#         # images.extend(seg_imgs)
#         sampled_idxes.extend(p)
#         if p + self.new_step < nimg:  #record['shot_info'][1]:
#             p += self.new_step
#     return sampled_idxes

# def prepare_test_imgs(self, idx):

#     width = result['ori_video_width']
#     height = result['ori_video_height']

#     # if proposal.shape[1] == 4:
#     #     proposal = proposal * np.array(
#     #         [width, height,
#     #             width, height])
#     # else:
#     #     proposal = proposal * np.array(
#     #         [width, height,
#     #             width, height, 1.0])

#     def prepare_single(img_group, scale, proposal):
#         (_img_group, img_shape, pad_shape, scale_factor,
#          crop_quadruple) = self.img_group_transform(
#              img_group,
#              scale,
#              crop_history=None,
#              flip=False,
#              keep_ratio=self.resize_keep_ratio)

#         _img_group = to_tensor(_img_group)
#         _img_meta = dict(
#             ori_shape=(height, width, 3),
#             img_shape=img_shape,
#             pad_shape=pad_shape,
#             scale_factor=scale_factor,
#             crop_quadruple=crop_quadruple,
#             flip=False)

#         proposal = proposal.astype(np.float32)
#         if proposal.shape[1] == 5:
#             proposal = proposal * np.array([width, height, width, height, 1.0])
#             score = proposal[:, 4, None]
#             proposal = proposal[:, :4]
#         else:
#             proposal = proposal * np.array([width, height, width, height])
#             score = None
#         _proposal = self.bbox_transform(
#             proposal, img_shape, scale_factor, flip, crop=crop_quadruple)
#         _proposal = np.hstack(
#             [_proposal, score] if score is not None else _proposal)
#         _proposal = to_tensor(_proposal)
#         return _img_group, _img_meta, _proposal

#     # indice = video_info['fps'] * \
#     #     (video_info['timestamp'] - _TIMESTAMP_START) + 1
#     indice = (len(imgs) - 1) // 2
#     # skip_offsets = np.random.randint(
#     #     self.new_step, size=self.old_length // self.new_step)

#     img_group = self._get_frames(indice, len(imgs))

#     scale = self.img_scale
#     _img_group, _img_meta, _proposal = prepare_single(img_group, scale, None,
#                                                       False, proposal)
#     _img_group = np.transpose(_img_group, (1, 0, 2, 3))
#     data['img_group_0'] = [_img_group]
#     data['img_meta'] = [DC(_img_meta, cpu_only=True)]
#     data['proposals'] = [_proposal]

#     return data
