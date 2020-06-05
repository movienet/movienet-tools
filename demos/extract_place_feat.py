import mmcv
import numpy as np

from mmmovie import PlaceExtractor

if __name__ == '__main__':
    weight_path = './model/resnet50_places365.pth'
    extractor = PlaceExtractor(weight_path, gpu=0)

    features = []
    for i in range(1, 4):
        img_path = './tests/data/test{:02d}.jpg'.format(i)
        img = mmcv.imread(img_path)
        output = extractor.extract(img)
        feature = output.detach().cpu().numpy().squeeze()
        feature /= np.linalg.norm(feature)
        features.append(feature)

    features = np.stack(features)
    confuse_matrix = features.dot(features.T)
    print('confusion matrixs of test01.jpg, test02.jpg and test03.jpg:')
    print(confuse_matrix)
