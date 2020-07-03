import os.path as osp

from .src import wav2stft_dist, wav2stft


class AudioExtractor(object):
    """Audio feature extractor class with options.
        audio: use STFT to extact audio feature
    """
    def __init__(self,):
        pass

    def extract(self, src_wave_path):
        stft = wav2stft(src_wave_path)
        return stft


class DistAudioExtractor(object):
    """Audio feature extractor class with options.
        audio: use STFT to extact audio feature

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

    def __init__(self,):
        pass

    def batch_extract(self,
                      audlist,
                      aud_prefix,
                      save_path,
                      num_workers=8,
                      replace_old=False):
        self.cfg = {}
        self.cfg['list_file'] = audlist
        self.cfg['replace_old'] = replace_old
        self.cfg['num_workers'] = num_workers
        self.cfg['aud_prefix'] = aud_prefix
        self.cfg['save_path'] = save_path
        wav2stft_dist(self.cfg)
