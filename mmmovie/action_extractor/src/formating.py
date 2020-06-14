import numpy as np
import torch
from mmcv.parallel import DataContainer as DC


class Collect(object):

    def __init__(self, keys, meta_keys=('filename', ), list_meta=True):
        self.keys = keys
        self.meta_keys = meta_keys
        self.list_meta = list_meta

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        if self.list_meta:
            data['img_meta'] = [DC(img_meta, cpu_only=True)]
        else:
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
        for key in results.keys():
            if key == 'img_meta':
                data['img_meta'] = [[
                    _meta.data for _meta in results['img_meta']
                ]]
            else:
                rst = results[key]
                if isinstance(rst, list):
                    data[key] = [
                        r.unsqueeze(dim=0).cuda(self.device) for r in rst
                    ]
                elif isinstance(rst, torch.Tensor):
                    data[key] = rst.unsqueeze(dim=0).cuda(self.device)
                else:
                    raise TypeError(
                        f"results[{key}] must be torch.Tensor or list")
        return data

    def __repr__(self):
        return self.__class__.__name__
