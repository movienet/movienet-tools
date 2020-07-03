import os.path as osp

from movienet.tools import ShotDetector

if __name__ == '__main__':
    sdt = ShotDetector(
        print_list=True,
        save_keyf=False,
        save_keyf_txt=True,
        split_video=False,
        begin_frame=0,
        end_frame=2000)

    video_path = osp.join('tests', 'data/test1.mp4')
    out_dir = osp.join('tests', 'data')
    sdt.shotdetect(video_path, out_dir)
