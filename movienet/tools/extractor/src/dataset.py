import os.path as osp

import mmcv
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from .transforms import CenterCrop, Normalize, OneImageCollate, Resize


class BaseDataProcessor(object):

    def __init__(self, gpu=0):
        self.pipeline = self.build_data_pipline(gpu)

    def __call__(self, img):
        """process an image.

        Args:
            img (np.array<uint8>): the input image, in BGR
        """
        return self.pipeline(img)

    def build_data_pipline(self, gpu):
        raise NotImplementedError


class PlaceDataProcessor(BaseDataProcessor):
    """image preprocess pipeline for place feature extractor."""

    def build_data_pipline(self, gpu):
        pipeline = Compose([
            Resize((256, 256)),
            CenterCrop((224, 224)),
            Normalize(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            ToTensor(),
            OneImageCollate(gpu)
        ])
        return pipeline


class PersonDataProcessor(BaseDataProcessor):
    """image preprocess pipeline for person feature extractor."""

    def build_data_pipline(self, gpu):
        pipeline = Compose([
            Resize((128, 256)),
            Normalize(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            ToTensor(),
            OneImageCollate(gpu)
        ])
        return pipeline


class FaceDataProcessor(BaseDataProcessor):
    """image preprocess pipeline for face feature extractor."""

    def build_data_pipline(self, gpu):
        pipeline = Compose([
            Resize((160, 160)),
            Normalize(
                mean=[127.5, 127.5, 127.5],
                std=[128.0, 128.0, 128.0],
                to_rgb=True),
            ToTensor(),
            OneImageCollate(gpu)
        ])
        return pipeline


class BaseDataset(Dataset):

    def __init__(self, img_list, img_prefix=None):
        if isinstance(img_list, list):
            self.img_list = img_list
        elif isinstance(img_list, str):
            assert osp.isfile(img_list)
            self.img_list = [x.strip() for x in open(img_list)]
        else:
            raise ValueError(
                'param "img_list" must be list or str, now it is {}'.format(
                    type(img_list)))
        self.img_prefix = img_prefix
        self.pipeline = self.build_data_pipline()

    def build_data_pipline(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        filename = osp.join(self.img_prefix, self.img_list[idx])
        img = mmcv.imread(filename)
        return self.pipeline(img)


class PersonDataset(BaseDataset):
    """Person dataset for extracting features."""

    def build_data_pipline(self):
        pipeline = Compose([
            Resize((256, 256)),
            CenterCrop((224, 224)),
            Normalize(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            ToTensor()
        ])
        return pipeline


class PlaceDataset(BaseDataset):
    """Place dataset for extracting features."""

    def build_data_pipline(self):
        pipeline = Compose([
            Resize((256, 256)),
            CenterCrop((224, 224)),
            Normalize(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            ToTensor()
        ])
        return pipeline


class FaceDataset(BaseDataset):
    """Face dataset for extracting features."""

    def build_data_pipline(self, gpu):
        pipeline = Compose([
            Resize((160, 160)),
            Normalize(
                mean=[127.5, 127.5, 127.5],
                std=[128.0, 128.0, 128.0],
                to_rgb=True),
            ToTensor()
        ])
        return pipeline
