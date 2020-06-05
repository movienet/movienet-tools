import os
import os.path as osp

import mmcv
import numpy as np

from mmmovie import FeatureExtractor, PlaceExtractor


class TestExtractor(object):

    @classmethod
    def setup_class(cls):
        cls.video_path = osp.join(osp.dirname(__file__), 'data/test.mp4')
        cls.img_list = [
            osp.join(osp.dirname(__file__), 'data/test{:02d}.jpg'.format(x))
            for x in range(1, 4)
        ]
        cls.out_dir = osp.join(osp.dirname(__file__), 'data')

    def test_extract_place_feat(self):
        weight = osp.join(os.getcwd(), 'model/resnet50_places365.pth')
        extractor = PlaceExtractor(weight, gpu=0)
        features = []
        for img_path in self.img_list:
            img = mmcv.imread(img_path)
            output = extractor.extract(img)
            feature = output.detach().cpu().numpy().squeeze()
            feature /= np.linalg.norm(feature)
            features.append(feature)

        features = np.stack(features)
        confuse_matrix = features.dot(features.T)
        assert int(confuse_matrix[0, 1] * 1000) == 633
        assert int(confuse_matrix[0, 2] * 1000) == 523
        assert int(confuse_matrix[1, 2] * 1000) == 880

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
