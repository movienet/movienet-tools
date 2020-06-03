import numpy as np
import torch
from mmcv.parallel import DataContainer as DC


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


class ToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


class ImageToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key].transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


class ToDataContainer(object):

    def __init__(self,
                 fields=(dict(key='img', stack=True), dict(key='gt_bboxes'),
                         dict(key='gt_labels'))):
        self.fields = fields

    def __call__(self, results):
        for field in self.fields:
            field = field.copy()
            key = field.pop('key')
            results[key] = DC(results[key], **field)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(fields={})'.format(self.fields)


class Collect(object):

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_meta'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, meta_keys={})'.format(
            self.keys, self.meta_keys)


class OneSampleCollate(object):

    def __init__(self, device=0):
        self.device = device

    def __call__(self, results):
        data = {}
        data['img_meta'] = [results['img_meta'].data]
        data['img'] = results['img'].unsqueeze(dim=0).cuda(self.device)
        return data

    def __repr__(self):
        return self.__class__.__name__
