import cv2
import numpy as np

from movienet.tools import PlaceExtractor

if __name__ == '__main__':
    weight_path = './model/resnet50_places365.pth'
    extractor = PlaceExtractor(weight_path, gpu=0)

    features = []
    img_list = []
    for i in range(1, 4):
        img_path = './tests/data/still{:02d}.jpg'.format(i)
        img = cv2.imread(img_path)
        feature = extractor.extract(img)
        feature /= np.linalg.norm(feature)
        features.append(feature)
        img_list.append(cv2.resize(img, (480, 200)))

    features = np.stack(features)
    confuse_matrix = features.dot(features.T)
    print('confusion matrixs of stills:')
    print(confuse_matrix)

    img_ref = cv2.putText(img_list[0], 'refer place', (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_1 = cv2.putText(img_list[1],
                        'similarity: {:.2f}'.format(confuse_matrix[0, 1]),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                        2)
    img_2 = cv2.putText(img_list[2],
                        'similarity: {:.2f}'.format(confuse_matrix[0, 2]),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                        2)
    img_show = cv2.hconcat((img_ref, img_1, img_2))
    cv2.imshow('place', img_show)
    cv2.waitKey()
