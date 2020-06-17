import time

import cv2
import numpy as np

from mmmovie import PersonDetector, PersonExtractor

if __name__ == '__main__':
    st = time.time()
    cfg = './model/cascade_rcnn_x101_64x4d_fpn.json'
    weight = './model/cascade_rcnn_x101_64x4d_fpn.pth'
    detector = PersonDetector('rcnn', cfg, weight)
    print('build model done: {:.2f}s'.format(time.time() - st))

    img_path = './tests/data/still01.jpg'
    img = cv2.imread(img_path)
    persons = detector.detect(img, show=True, conf_thr=0.9)
    assert persons.shape[0] == 2
    person_imgs = detector.crop_person(img, persons)

    print('{} persons detected!'.format(persons.shape[0]))
    print('persons:')
    print(persons)

    weight_path = './model/resnet50_csm.pth'
    extractor = PersonExtractor(weight_path, gpu=0)

    img_rose = cv2.imread('./tests/data/body_rose.jpg')
    feat_rose = extractor.extract(img_rose)
    feat_rose /= np.linalg.norm(feat_rose)

    img_jack = cv2.imread('./tests/data/body_jack.jpg')
    feat_jack = extractor.extract(img_jack)
    feat_jack /= np.linalg.norm(feat_jack)
    predicts = []
    for i in range(2):
        feat = extractor.extract(person_imgs[i])
        feat /= np.linalg.norm(feat)
        prob_jack = feat[np.newaxis, ...].dot(feat_jack)[0]
        prob_rose = feat[np.newaxis, ...].dot(feat_rose)[0]
        print('person {}: {:.2f} vs. {:.2f}'.format(i, prob_jack, prob_rose))
        if prob_jack > prob_rose:
            predicts.append('Jack | {:.2f}'.format(prob_jack))
        else:
            predicts.append('Rose | {:.2f}'.format(prob_rose))

    # draw and show
    for i in range(2):
        img = cv2.rectangle(img, (int(persons[i, 0]), int(persons[i, 1])),
                            (int(persons[i, 2]), int(persons[i, 3])),
                            (0, 255, 0), 2)
        img = cv2.putText(img, predicts[i],
                          (int(persons[i, 0] + 10), int(persons[i, 1] + 30)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    h = img.shape[0]
    img_jack = cv2.resize(img_jack, (h // 2, h // 2))
    img_jack = cv2.putText(img_jack, 'refer body: Jack', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    img_rose = cv2.resize(img_rose, (h // 2, h // 2))
    img_rose = cv2.putText(img_rose, 'refer body: Rose', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    refer_img = cv2.vconcat((img_jack, img_rose))
    refer_img = cv2.resize(refer_img, (refer_img.shape[1], h))
    img_show = cv2.hconcat((refer_img, img))
    cv2.imshow('person', img_show)
    cv2.waitKey()
