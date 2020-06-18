import argparse
import multiprocessing
import os
import os.path as osp
import subprocess
from datetime import datetime

import librosa
import numpy as np

global parallel_cnt
global parallel_num
parallel_cnt = 0


def call_back(rst):
    global parallel_cnt
    global parallel_num
    parallel_cnt += 1
    if parallel_cnt % 100 == 0:
        print('{}, {:5d} / {:5d} done!'.format(datetime.now(), parallel_cnt,
                                               parallel_num))


def run_mp42wav(args, file_item_mp4):
    file_item = file_item_mp4.split('.m')[0]
    src_video_fn = osp.join(args.src_video_path, '{}'.format(file_item_mp4))
    dst_video_fn = osp.join(args.save_path, '{}.wav'.format(file_item))
    if not args.replace_old and osp.exists(dst_video_fn):
        return 0
    call_list = ['ffmpeg']
    call_list += ['-v', 'quiet']
    call_list += ['-i', src_video_fn, '-f', 'wav']
    call_list += ['-map_chapters', '-1']  # remove meta stream
    call_list += [dst_video_fn]
    subprocess.call(call_list)
    if not osp.exists(dst_video_fn):
        wav_np = np.zeros((16000 * 4), np.float32)
        librosa.output.write_wav(dst_video_fn, wav_np, sr=16000)
        print(file_item_mp4, 'not exist, and zero is written into it instead')


def run(args, file_item_mp4):
    run_mp42wav(args, file_item_mp4)


def main(args):
    os.makedirs(args.save_path, exist_ok=True)

    if args.listfile is None:
        video_list = sorted(os.listdir(args.src_video_path))
    else:
        video_list = [x.strip() for x in open(args.listfile)]

    flag_folder = False
    if osp.isdir(osp.join(args.src_video_path, video_list[0])):
        flag_folder = True
    elif osp.isfile(osp.join(args.src_video_path, video_list[0])):
        pass

    global parallel_num
    parallel_num = 0
    if flag_folder:
        file_list = []
        for video_id in video_list:
            shot_id_mp4_list = sorted(
                os.listdir(osp.join(args.src_video_path, video_id)))
            for shot_id_mp4 in shot_id_mp4_list:
                print(osp.join(args.src_video_path, video_id, shot_id_mp4))
                if osp.isdir(
                        osp.join(args.src_video_path, video_id, shot_id_mp4)):
                    raise RuntimeError('There is subfolder in {}, \
                                    but currently the program only supports \
                                    two-level folder'.format(
                        osp.join(args.src_video_path, video_id, shot_id_mp4)))
                file_list.append(osp.join(video_id, shot_id_mp4))
                parallel_num += 1
    else:
        parallel_num = len(video_list)
        file_list = video_list
    if args.num_workers > 1:
        pool = multiprocessing.Pool(processes=args.num_workers)
    for file_item_mp4 in file_list:
        if '/' in file_item_mp4:
            video_id = file_item_mp4.split('/')[0]
            os.makedirs(osp.join(args.save_path, video_id), exist_ok=True)
        if args.num_workers == 1:
            run(args, file_item_mp4)
        elif args.num_workers > 1:
            pool.apply_async(
                run, (args, file_item_mp4), callback=call_back)
    if args.num_workers > 1:
        pool.close()
        pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nw', '--num_workers', type=int,
                        default=8, help='number of processors.')
    parser.add_argument('--listfile', type=str)
    parser.add_argument('--src_video_path', type=str)
    parser.add_argument('--save_path',  type=str)
    parser.add_argument('--replace_old', action="store_true")
    args = parser.parse_args()
    main(args)
