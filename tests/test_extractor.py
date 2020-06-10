import os
import os.path as osp

import cv2

from mmmovie import (FaceExtractor, FeatureExtractor, PersonExtractor,
                     PlaceExtractor)


class TestExtractor(object):

    @classmethod
    def setup_class(cls):
        cls.video_path = osp.join(osp.dirname(__file__), 'data/test.mp4')
        cls.face_img_path = osp.join(
            osp.dirname(__file__), 'data/face_rose.jpg')
        cls.person_img_path = osp.join(
            osp.dirname(__file__), 'data/body_rose.jpg')
        cls.still_path = osp.join(osp.dirname(__file__), 'data/still01.jpg')

        cls.out_dir = osp.join(osp.dirname(__file__), 'data')

    def test_extract_place_feat(self):
        weight = osp.join(os.getcwd(), 'model/resnet50_places365.pth')
        extractor = PlaceExtractor(weight, gpu=0)
        assert extractor is not None
        img = cv2.imread(self.still_path)
        feature = extractor.extract(img)
        assert int(feature.sum() * 100) == 64151

    def test_extract_person_feat(self):
        weight = osp.join(os.getcwd(), 'model/resnet50_csm.pth')
        extractor = PersonExtractor(weight, gpu=0)
        assert extractor is not None
        img = cv2.imread(self.person_img_path)
        feature = extractor.extract(img)
        assert int(feature.sum() * 100) == -2612

    def test_extract_face_feat(self):
        weight = osp.join(os.getcwd(), 'model/irv1_vggface2.pth')
        extractor = FaceExtractor(weight, gpu=0)
        assert extractor is not None
        img = cv2.imread(self.face_img_path)
        feature = extractor.extract(img)
        assert int(feature.sum() * 100) == -222

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
