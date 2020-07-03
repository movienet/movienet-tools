import argparse
import os.path as osp

from movienet.tools import DistAudioExtractor


def main(args):
    extractor = DistAudioExtractor()

    assert osp.isfile(args.listfile)
    assert osp.isdir(args.aud_prefix)
    extractor.batch_extract(
        args.listfile,
        args.aud_prefix,
        args.save_path,
        args.num_workers,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Audio Features')
    parser.add_argument(
        '--listfile',
        type=str,
        default=None,
        help='the list file of the audio names')
    parser.add_argument(
        '--aud_prefix', type=str, default=None, help='prefix of the audios')
    parser.add_argument(
        '--save_path', type=str, default=None, help='save path')
    parser.add_argument(
        '--num_workers', type=int, default=8, help='number of workers for cpu')
    args = parser.parse_args()
    main(args)
