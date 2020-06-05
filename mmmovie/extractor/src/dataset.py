import os.path as osp

import mmcv
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from .transforms import CenterCrop, Normalize, OneImageCollate, Resize


class DataProcessor(object):
    """image preprocess pipeline."""

    def __init__(self, gpu=0):
        self.pipeline = Compose([
            Resize((256, 256)),
            CenterCrop((224, 224)),
            Normalize(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            ToTensor(),
            OneImageCollate(gpu)
        ])

    def __call__(self, img):
        """process an image.

        Args:
            img (np.array<uint8>): the input image, in BGR
        """
        return self.pipeline(img)


class CustomDataset(Dataset):
    """Custom dataset for detection."""

    def __init__(self, img_list, img_prefix=None, gpu=0):
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
        self.data_processor = DataProcessor(gpu)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self.prepare_test_img(idx)

    def prepare_test_img(self, idx):
        filename = osp.join(self.img_prefix, self.img_list[idx])
        img = mmcv.imread(filename)
        return self.data_processor(img)
