import json
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint

from .facedet.dataset import FaceDataProcessor
from .facedet.mtcnn import MTCNN

resources_dir = osp.join(osp.dirname(__file__), '../../../model')


class FaceDetector(object):

    def __init__(self,
                 cfg_path=osp.join(resources_dir, 'mtcnn.json'),
                 weight_path=osp.join(resources_dir, 'mtcnn.pth'),
                 gpu=0):
        self.model = self.build_mtcnn(cfg_path, weight_path)
        self.model.eval()
        self.model.cuda(gpu)
        self.data_processor = FaceDataProcessor(gpu)

    def build_mtcnn(self, cfg_path, weight_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        model = MTCNN(**cfg)
        load_checkpoint(model, weight_path, map_location='cpu')
        return model

    def detect(self, img, conf_thr=0.5, show=False):
        assert conf_thr >= 0 and conf_thr < 1
        if isinstance(img, str):
            filename = img
            assert osp.isfile(filename)
            img = mmcv.imread(filename)
        data = self.data_processor(img)
        with torch.no_grad():
            results = self.model(data)
        faces = results[0][0]
        landmarks = results[1][0]
        keep_idx = faces[:, -1] > conf_thr
        faces = faces[keep_idx]
        landmarks = landmarks[keep_idx]
        if show:
            img_show = self.draw_face(img.copy(), faces, landmarks)
            cv2.imshow('face', img_show)
            cv2.waitKey()
        return faces, landmarks

    def draw_face(self, img, faces, landmarks=None):
        num_faces = faces.shape[0]
        for i in range(num_faces):
            img = cv2.rectangle(img, (int(faces[i, 0]), int(faces[i, 1])),
                                (int(faces[i, 2]), int(faces[i, 3])),
                                (0, 255, 0), 2)
            if landmarks is not None:
                for j in range(5):
                    img = cv2.circle(
                        img,
                        (int(landmarks[i, j, 0]), int(landmarks[i, j, 1])), 2,
                        (0, 255, 0), 2)
            img = cv2.putText(img, 'prob:{:.2f}'.format(faces[i, -1]),
                              (int(faces[i, 0]), int(faces[i, 1] - 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img

    def crop_face(self,
                  img,
                  faces,
                  save_dir=None,
                  save_prefix=None,
                  img_scale=160):
        num_faces = faces.shape[0]
        h, w = img.shape[:2]
        face_list = []
        for i in range(num_faces):
            x1, y1, x2, y2 = faces[i, :4].astype(np.int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            face_img = img[y1:y2, x1:x2]
            if img_scale is not None:
                face_img = cv2.resize(face_img, (img_scale, img_scale))
            if save_dir is not None and save_prefix is not None:
                save_path = osp.join(save_dir,
                                     save_prefix + '_{:02d}.jpg'.format(i))
                cv2.imwrite(save_path, face_img)
            face_list.append(face_img)
        return face_list
