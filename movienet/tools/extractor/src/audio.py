"""Audio extractor, supports folder and video input folders.

xxxx0\nxxxx1\nxxxx2\n  # folders of mp4 video videos
xxxx0.mp4\nxxxx1.mp4\nxxxx2.mp4\n or
xxxx0/xxxx0.mp4\nxxxxx0/xxxx1.mp4\nxxxx1/xxxx0.mp4\nxxxx1/xxxx1.mp4\n
"""
import multiprocessing
import os
import os.path as osp
import subprocess
from datetime import datetime

import librosa
import mmcv
import numpy as np
import pdb


def wav2stft(src_wav_path):
    k = 3  # sample episode num
    time_unit = 3  # unit: second
    file_item_mp4 = src_wav_path.split("/")[-1]
    data, fs = librosa.core.load(src_wav_path, sr=16000)
    # normalize
    mean = (data.max() + data.min()) / 2
    span = (data.max() - data.min()) / 2
    eps = 1e-6
    if span < eps:
        span = 1
    data = (data - mean) / span  # range: [-1,1]

    D = librosa.core.stft(data, n_fft=512)
    freq = np.abs(D)
    stft = librosa.core.amplitude_to_db(freq)

    # tile
    rate = stft.shape[1] / (len(data) / fs)
    thr = int(np.ceil(time_unit * rate / k * (k + 1)))
    stft_copy = stft.copy()
    while stft.shape[1] < thr:
        tmp = stft_copy.copy()
        stft = np.concatenate((stft, tmp), axis=1)

    if stft.shape[1] <= 90:  # hard code in audio feat extractor
        raise RuntimeError('the shape of {} is wrong {}'.format(
            file_item_mp4, stft.shape))

    # sample
    n = stft.shape[1]
    milestone = [x[0] for x in np.array_split(np.arange(n), k + 1)[1:]]
    span = 15  # hard code in audio feat extractor
    stft_img = []
    for i in range(k):
        stft_img.append(stft[:, milestone[i] - span:milestone[i] + span])
    stft = np.concatenate(stft_img, axis=1)
    if stft.shape[1] != 90:
        raise RuntimeError('the shape of {} is wrong {}'.format(
            file_item_mp4, stft.shape))
    return stft


def call_back(rst):
    global parallel_cnt
    global parallel_num
    parallel_cnt += 1
    if parallel_cnt % 100 == 0:
        print('{}, {:5d} / {:5d} done!'.format(datetime.now(), parallel_cnt,
                                               parallel_num))


def run(cfg_dict, file_item_mp4):
    cfg = mmcv.Config(cfg_dict)
    freq = wav2stft(osp.join(cfg.aud_prefix, file_item_mp4))
    file_item = file_item_mp4.split('.w')[0]
    feat_path = osp.join(cfg.save_path, '{}.npy'.format(file_item))
    np.save(feat_path, freq)


def wav2stft_dist(cfg):
    cfg_dict = cfg.copy()
    cfg = mmcv.Config(cfg_dict)
    print(cfg)
    os.makedirs(cfg.save_path, exist_ok=True)

    if cfg.list_file is None:
        video_list = sorted(os.listdir(cfg.aud_prefix))
    else:
        video_list = [x.strip() for x in open(cfg.list_file)]

    flag_folder = False
    if osp.isdir(osp.join(cfg.aud_prefix, video_list[0])):
        flag_folder = True
    elif osp.isfile(osp.join(cfg.aud_prefix, video_list[0])):
        pass
    global parallel_num
    global parallel_cnt
    parallel_num = 0
    parallel_cnt = 0
    if flag_folder:
        file_list = []
        for video_id in video_list:
            shot_id_mp4_list = sorted(
                os.listdir(osp.join(cfg.aud_prefix, video_id)))
            for shot_id_mp4 in shot_id_mp4_list:
                print(osp.join(cfg.aud_prefix, video_id, shot_id_mp4))
                if osp.isdir(
                        osp.join(cfg.aud_prefix, video_id, shot_id_mp4)):
                    raise RuntimeError('There is subfolder in {}, \
                                    but currently the program only supports \
                                    two-level folder'.format(
                        osp.join(cfg.aud_prefix, video_id, shot_id_mp4)))
                file_list.append(osp.join(video_id, shot_id_mp4))
                parallel_num += 1
    else:
        parallel_num = len(video_list)
        file_list = video_list
    if cfg.num_workers > 1:
        pool = multiprocessing.Pool(processes=cfg.num_workers)
    for file_item_mp4 in file_list:
        if '/' in file_item_mp4:
            video_id = file_item_mp4.split('/')[0]
            os.makedirs(osp.join(cfg.save_path, video_id), exist_ok=True)
        if cfg.num_workers == 1:
            run(cfg_dict, file_item_mp4)
        elif cfg.num_workers > 1:
            pool.apply_async(
                run, (cfg_dict, file_item_mp4), callback=call_back)
    if cfg.num_workers > 1:
        pool.close()
        pool.join()
