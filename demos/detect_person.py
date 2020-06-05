import time

import mmcv

from mmmovie import PersonDetector

if __name__ == '__main__':
    st = time.time()
    cfg = './model/cascade_rcnn_x101_64x4d_fpn.json'
    weight = './model/cascade_rcnn_x101_64x4d_fpn.pth'
    detector = PersonDetector('rcnn', cfg, weight)
    print('build model done: {:.2f}s'.format(time.time() - st))

    img_path = './tests/data/test.jpg'
    img = mmcv.imread(img_path)
    result = detector.detect(img, show=True)

    print('{} faces detected!'.format(result.shape[0]))
    print('faces:')
    print(result)
