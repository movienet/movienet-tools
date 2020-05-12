import os
import os.path as osp
import sys

import pytest

from mmmovie import FeatureExtractor


class TestExtractor(object):

    @classmethod
    def setup_class(cls):
        cls.video_path = osp.join(osp.dirname(__file__), 'data/test.mp4')
        cls.out_dir = osp.join(osp.dirname(__file__), 'data')

    # @staticmethod
    # def test_extract_place_feat_folder():
    #     extractor = FeatureExtractor(mode="place", data_root=osp.join(osp.dirname(__file__), 'data'),
    #                                  list_file=osp.join(osp.dirname(__file__), 'data/meta/list_test_folder.txt'))
    #     extractor.extract()
    #     assert len(os.listdir(osp.join(osp.dirname(__file__), 'data/place_feat/test1'))) > 3

    # @staticmethod
    # def test_extract_place_feat_video():
    #     extractor = FeatureExtractor(mode="place", data_root=osp.join(osp.dirname(__file__), 'data'),
    #                                  list_file=osp.join(osp.dirname(__file__), 'data/meta/list_test_video.txt'))
    #     extractor.extract()
    #     assert len(os.listdir(osp.join(osp.dirname(__file__), 'data/place_feat/test1'))) > 3

    # @staticmethod
    # def test_extract_place_feat_image():
    #     extractor = FeatureExtractor(mode="place", data_root=osp.join(osp.dirname(__file__), 'data'),
    #                                  list_file=osp.join(osp.dirname(__file__), 'data/meta/list_test_image.txt'))
    #     extractor.extract()
    #     assert len(os.listdir(osp.join(osp.dirname(__file__), 'data/place_feat/test1'))) > 3

    @staticmethod
    def test_extract_audio_feat_folder():
        extractor = FeatureExtractor(
            mode='audio',
            data_root=osp.join(osp.dirname(__file__), 'data'),
            list_file=osp.join(
                osp.dirname(__file__), 'data/meta/list_test_folder.txt'))
        extractor.extract()
        assert len(
            os.listdir(osp.join(osp.dirname(__file__),
                                'data/aud_feat/test1'))) == 3
        assert len(
            os.listdir(osp.join(osp.dirname(__file__),
                                'data/aud_feat/test2'))) == 3

    @staticmethod
    def test_extract_audio_feat_video1():
        extractor = FeatureExtractor(
            mode='audio',
            data_root=osp.join(osp.dirname(__file__), 'data'),
            list_file=osp.join(
                osp.dirname(__file__), 'data/meta/list_test_video1.txt'))
        extractor.extract()
        assert len(
            os.listdir(osp.join(osp.dirname(__file__),
                                'data/aud_feat/test1'))) == 3

    @staticmethod
    def test_extract_audio_feat_video2():
        extractor = FeatureExtractor(
            mode='audio',
            data_root=osp.join(osp.dirname(__file__), 'data'),
            src_video_path='shot_split_video/test2',
            dst_wav_path='aud_wav/test2',
            dst_stft_path='aud_feat/test2',
            list_file=osp.join(
                osp.dirname(__file__), 'data/meta/list_test_video2.txt'))
        extractor.extract()
        assert len(
            os.listdir(osp.join(osp.dirname(__file__),
                                'data/aud_feat/test2'))) == 3
