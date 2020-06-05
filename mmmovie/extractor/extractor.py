import os.path as osp

from .src import extract_audio_feat


class FeatureExtractor(object):
    """Feature extractor class with options.
        place: use ImageNet pretrained ResNet50 to extract image feature
        audio: use STFT to extact audio feature
        action: TBD
        cast: TBD

    Args:
        list_file: The list of folders/videos/images to be processed,\
            in the form of
            folders video0\nvideo1\nvideo2\n
            videos video0.mp4\nvideo1.mp4\nvideo2.mp4\n
                or video0/shot0.mp4\nvideo0/shot1.mp4\n
                   video1/shot0.mp4\nvideo1/shot1.mp4\n
            images video0/shot0.jpg\nvideo0/shot1.jpg\n
                   video1/shot0.jpg\nvideo1/shot1.jpg\n
            folders xxxx0\nxxxx1\nxxxx2\n
            videos xxxx0.mp4\nxxxx1.mp4\nxxxx2.mp4\n
                or xxxx0/xxxx0.mp4\nxxxxx0/xxxx1.mp4\n
                   xxxx1/xxxx0.mp4\nxxxx1/xxxx1.mp4\n
            images xxxx0/xxxx0.jpg\nxxxxx0/xxxx1.jpg\nxxxx1/xxxx0.jpg
    """

    def __init__(
        self,
        # for common
        mode='audio',
        data_root='/mnt/SSD/ayrao/mmmovie/test/data',
        list_file='/mnt/SSD/ayrao/mmmovie/test/data/meta/list_test.txt',
        # for place
        batch_size=4,
        save_one_frame_feat=False,
        # for audio
        num_workers=8,
        replace_old=False,
        src_video_path=None,  # default to shot_split_video
        dst_wav_path=None,  # default to aud_wav
        dst_stft_path=None,  # default to aud_stft
    ):
        assert mode in ['place', 'action', 'cast', 'audio']
        self.mode = mode
        self.cfg = {}
        if mode == 'audio':
            self.cfg['data_root'] = data_root
            self.cfg['list_file'] = list_file
            self.cfg['replace_old'] = replace_old
            self.cfg['num_workers'] = num_workers
            self.cfg['src_video_path'] = osp.join(
                data_root,
                'shot_split_video') if src_video_path is None else osp.join(
                    data_root, src_video_path)
            self.cfg['dst_wav_path'] = osp.join(
                data_root, 'aud_wav') if dst_wav_path is None else osp.join(
                    data_root, dst_wav_path)
            self.cfg['dst_stft_path'] = osp.join(
                data_root, 'aud_feat') if dst_stft_path is None else osp.join(
                    data_root, dst_stft_path)

    def extract(self, ):
        print('*' * 20 + ' Extract {} feat '.format(self.mode) + '*' * 20)
        if self.mode == 'audio':
            extract_audio_feat(self.cfg)
