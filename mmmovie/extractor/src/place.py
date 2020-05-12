"""Place extractor supports folder input folders.

xxxx0\nxxxx1\nxxxx2\n  # folders of jpg image.
"""
from __future__ import absolute_import, print_function
import os
import os.path as osp
import pickle
import time
from collections import OrderedDict
from datetime import datetime

import mmcv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError('Cannot convert {} to numpy array'.format(
            type(tensor)))
    return tensor


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ResNet50(torch.nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        self.base = models.resnet50(pretrained=pretrained)

    def forward(self, x):
        for name, module in self.base._modules.items():
            x = module(x)
            # print(name, x.size())
            if name == 'avgpool':
                x = x.view(x.size(0), -1)
                feature = x.clone()
        return feature, x


class Extractor(object):

    def __init__(self, model):
        super(Extractor, self).__init__()
        self.model = model
        # pprint(self.model.module)

    def extract_feature(self, data_loader, print_summary=True):
        self.model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        features = OrderedDict()
        scores = OrderedDict()

        end = time.time()
        for i, (imgs, fnames) in enumerate(data_loader):
            data_time.update(time.time() - end)
            outputs = self.model(imgs)

            for fname, feat, score in zip(fnames, outputs[0], outputs[1]):
                features[fname] = feat.cpu().data
                scores[fname] = score.cpu().data

            batch_time.update(time.time() - end)
            end = time.time()

            if print_summary:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'.format(i + 1, len(data_loader),
                                                      batch_time.val,
                                                      batch_time.avg,
                                                      data_time.val,
                                                      data_time.avg))
        return features, scores


class Preprocessor(object):

    def __init__(self, dataset, images_path, default_size, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.images_path = images_path
        self.transform = transform
        self.default_size = default_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname = self.dataset[index]
        fpath = osp.join(self.images_path, fname)
        if os.path.isfile(fpath):
            img = Image.open(fpath).convert('RGB')
        else:
            img = Image.new('RGB', self.default_size)
            print('No such images: {:s}'.format(fpath))
        if self.transform is not None:
            img = self.transform(img)
        return img, fname


def get_data(video_id, src_img_path, batch_size, workers):

    dataset = os.listdir(src_img_path)
    if len(dataset) % batch_size < 8:
        for i in range(8 - len(dataset) % batch_size):
            dataset.append(dataset[-1])

    # data transforms
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalizer,
    ])

    # data loaders
    data_loader = DataLoader(
        Preprocessor(
            dataset,
            src_img_path,
            default_size=(256, 256),
            transform=data_transformer),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True)

    return dataset, data_loader


def get_img_folder(data_root, video_id):
    img_folder = osp.join(data_root, video_id)
    if osp.isdir(img_folder):
        return img_folder
    else:
        print('No such movie: {}'.format(video_id))
        return None


def extract_place_feat(cfg):
    # conver the cfg dict to mmcv.Config
    cfg = mmcv.Config(cfg)
    print(cfg)
    cudnn.benchmark = True
    # create model
    model = ResNet50(pretrained=True)
    model = torch.nn.DataParallel(model).cuda()
    # create and extractor
    extractor = Extractor(model)

    if cfg.list_file is None:
        video_list = sorted(os.listdir(cfg.src_img_path))
    else:
        video_list = [x.strip() for x in open(cfg.list_file)]
    video_list = [i.split('.m')[0] for i in video_list
                  ]  # to remove suffix .mp4 .mov etc. if applicable
    video_list = video_list[cfg.st:cfg.ed]
    print('****** Total {} videos ******'.format(len(video_list)))

    for idx_m, video_id in enumerate(video_list):
        print('****** {}, {} / {}, {} ******'.format(datetime.now(), idx_m + 1,
                                                     len(video_list),
                                                     video_id))
        dst_path = osp.join(cfg.dst_path, video_id)
        os.makedirs(dst_path, exist_ok=True)
        src_img_path = get_img_folder(cfg.src_img_path, video_id)
        if not osp.isdir(src_img_path):
            print('Cannot find images!')

        feat_dst_name = osp.join(dst_path, 'feat.pkl')
        score_dst_name = osp.join(dst_path, 'score.pkl')
        if osp.isfile(feat_dst_name) and osp.isfile(score_dst_name):
            print('{}, {} exist.'.format(datetime.now(), video_id))
            continue
        # create data loaders
        dataset, data_loader = get_data(video_id, src_img_path, cfg.batch_size,
                                        cfg.workers)

        # extract feature
        try:
            print('{}, extracting features...'.format(datetime.now()))
            feat_dict, score_dict = extractor.extract_feature(
                data_loader, print_summary=False)
            for key, item in feat_dict.items():
                item = to_numpy(item)
                os.makedirs(
                    osp.join(cfg.dst_feat_path, video_id), exist_ok=True)
                img_ind = key.split('_')[-1].split('.jpg')[0]
                if cfg.save_one_frame_feat:
                    if img_ind == '1':  # for 3 images 1 shot only
                        shot_ind = key.split('_')[1]
                        dst_fn = osp.join(cfg.dst_feat_path, video_id,
                                          'shot_{}.npy'.format(shot_ind))
                        np.save(dst_fn, item)
                    else:
                        continue
                else:
                    dst_fn = osp.join(cfg.dst_feat_path, video_id,
                                      '{}.npy'.format(key.split('.jpg')[0]))
                    np.save(dst_fn, item)

            print('{}, saving...'.format(datetime.now()))
            with open(feat_dst_name, 'wb') as f:
                pickle.dump(feat_dict, f)
            with open(score_dst_name, 'wb') as f:
                pickle.dump(score_dict, f)
        except Exception as e:
            print('{} error! {}'.format(video_id, e))
        print('\n')
