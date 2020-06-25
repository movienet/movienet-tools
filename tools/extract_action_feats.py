import argparse
import os
import os.path as osp
import mmcv
from mmmovie.utils import read_movie_list
import tempfile
# from mmmovie.action_extractor.src.dataset import ActionDataset
from mmmovie.action_extractor.src.video import VideoFileBackend
from mmmovie.detector.parallel_persondetector import ParallelPersonDetector
from mmmovie.action_extractor.action_extract_manager import ActionExtractManager
from mmmovie.action_extractor.action_extractor import ParallelActionExtractor


def main(args):
    movie_ids = read_movie_list(args.movie_list)
    manager = ActionExtractManager()
    if args.detect:
        assert not args.tracklet_dir
        if args.temp_dir is None:
            args.temp_dir = tempfile.mkdtemp()
        args.tracklet_dir = args.temp_dir
        mmcv.mkdir_or_exist(args.temp_dir)
        detector = ParallelPersonDetector(
            args.arch,
            args.detector_cfg,
            args.detector_weight,
            gpu_ids=list(range(args.ngpu)))
        for movie_id in movie_ids:
            shot_file = osp.join(args.movienet_root, 'shot', f"{movie_id}.txt")
            video = VideoFileBackend(
                'twolevel',
                osp.join(args.movienet_root, 'frame', movie_id),
                shot_file=shot_file)
            tracklets = manager.run_detect(
                detector, video, shot_file, imgs_per_gpu=args.imgs_per_gpu)
            tracklet_file = osp.join(args.temp_dir, f"{movie_id}.pkl")
            mmcv.dump(tracklets, tracklet_file)

    extractor = ParallelActionExtractor(args.extractor_cfg,
                                        args.extractor_weight)
    for movie_id in movie_ids:
        shot_file = osp.join(args.movienet_root, 'shot', f"{movie_id}.txt")
        video = VideoFileBackend(
            'twolevel',
            osp.join(args.movienet_root, 'frame', movie_id),
            shot_file=shot_file)
        tracklet_file = osp.join(args.tracklet_dir, f"{movie_id}.pkl")
        result = manager.run_extract(extractor, video, shot_file,
                                     tracklet_file)
        mmcv.dump(result, osp.join(args.save_dir, f"{movie_id}.pkl"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Action Feature')
    parser.add_argument('movie_list', type=str)
    parser.add_argument('save_dir', type=str)
    parser.add_argument('movienet_root', type=str, help="movienet data root")
    parser.add_argument('--imgs_per_gpu', type=int, default=1)
    parser.add_argument(
        '--detector_cfg',
        type=str,
        default=osp.join(os.getcwd(),
                         'model/cascade_rcnn_x101_64x4d_fpn.json'))
    parser.add_argument(
        '--extractor_cfg',
        type=str,
        default=osp.join(
            os.getcwd(),
            'model/ava_fast_rcnn_nl_r50_c4_1x_kinetics_pretrain_crop.py'))
    parser.add_argument(
        '--arch',
        type=str,
        choices=['rcnn', 'retina'],
        default='rcnn',
        help='architechture of the detector')
    parser.add_argument(
        '--detector_weight',
        type=str,
        default=osp.join(os.getcwd(), 'model/cascade_rcnn_x101_64x4d_fpn.pth'),
        help='the weight of the model')
    parser.add_argument(
        '--extractor_weight',
        type=str,
        default=osp.join(
            os.getcwd(),
            'model/ava_fast_rcnn_nl_r50_c4_1x_kinetics_pretrain_crop.pth'),
        help='the weight of the model')
    parser.add_argument('--tracklet_dir', default=None, type=str)
    parser.add_argument('--temp_dir', default=None, type=str)
    parser.add_argument('--ngpu', type=int, default=1)

    parser.add_argument('--detect', action='store_true')

    args = parser.parse_args()

    main(args)