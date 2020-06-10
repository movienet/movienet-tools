import mmcv
import numpy as np

from mmmovie import PersonExtractor

if __name__ == '__main__':
    weight_path = './model/resnet50_csm.pth'
    extractor = PersonExtractor(weight_path, gpu=0)

    features = []
    for i in range(1, 4):
        img_path = './tests/data/person{:02d}.jpg'.format(i)
        img = mmcv.imread(img_path)
        feature = extractor.extract(img)
        feature /= np.linalg.norm(feature)
        features.append(feature)

    features = np.stack(features)
    confuse_matrix = features.dot(features.T)
    print('confusion matrixs of person01.jpg, person02.jpg and person03.jpg:')
    print(confuse_matrix)
