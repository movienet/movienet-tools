import time

import cv2
import mmcv

from mmmovie import FaceDetector


def draw_face(img, faces, landmarks):
    num_faces = faces.shape[0]
    for i in range(num_faces):
        img = cv2.rectangle(img, (int(faces[i, 0]), int(faces[i, 1])),
                            (int(faces[i, 2]), int(faces[i, 3])), (0, 255, 0),
                            2)
        for j in range(5):
            img = cv2.circle(
                img, (int(landmarks[i, j, 0]), int(landmarks[i, j, 1])), 2,
                (0, 255, 0), 2)
    return img


if __name__ == '__main__':
    st = time.time()
    cfg = './model/mtcnn.json'
    weight = './model/mtcnn.pth'
    detector = FaceDetector(cfg, weight)
    print('build model done: {:.2f}s'.format(time.time() - st))

    img_path = './tests/data/test01.jpg'
    img = mmcv.imread(img_path)
    faces, landmarks = detector.detect(img)

    num_faces = faces.shape[0]
    print('{} faces detected!'.format(num_faces))
    print('bboxes & confidence:')
    print(faces)
    print('landmarks:')
    print(landmarks)

    img = draw_face(img, faces, landmarks)
    cv2.imshow('haha', img)
    cv2.waitKey()
