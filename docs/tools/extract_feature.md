### Face Detector and Feature Extractor

The face detector and Feature Extractor here are modified from [facenet-pytorch](https://github.com/timesler/facenet-pytorch). The detection model is MTCNN and the feature extraction model is InceptionResnetV1.

Here is an example about how to detect a face and represent it as a feature vector.

```python
import cv2
from movienet.tools import FaceDetector, FaceExtractor

## build models
cfg = './model/mtcnn.json'
weight_det = './model/mtcnn.pth'
detector = FaceDetector(cfg, weight_det)

weight_ext = './model/irv1_vggface2.pth'
extractor = FaceExtractor(weight_ext, gpu=0)

## read image
img_path = './tests/data/portrait_jack.jpg'
img = cv2.imread(img_path)

## detect and crop face
faces, _ = detector.detect(img)
face_imgs = detector.crop_face(img, faces)

## extractor face feature
feat = extractor.extract(face_imgs[0])
```

#### Distributed inferring (under test)

```
bash tools/dist_infer.sh scripts/detect_face.py 8
```

### Person Detector & Feature Extractor

The person detector here is a casecadercnn trained with the bbox annotations in MovieNet based on the detection codebase [MMDetection](https://github.com/open-mmlab/mmdetection). The person Feature Extractor here is a resnet50 trained with the cast annotations in MovieNet.

Here is an example about how to detect a person and represent it as a feature vector, which is quite similar to the demo of face detection and extraction.

```python
import cv2
from movienet.tools import PersonDetector, PersonExtractor

# build models
cfg = './model/cascade_rcnn_x101_64x4d_fpn.json'
weight_det = './model/cascade_rcnn_x101_64x4d_fpn.pth'
detector = PersonDetector('rcnn', cfg, weight_det)
weight_ext = './model/resnet50_csm.pth'
extractor = PersonExtractor(weight_ext, gpu=0)

# detect and crop person
img_path = './tests/data/still01.jpg'
img = cv2.imread(img_path)
persons = detector.detect(img, show=True, conf_thr=0.9)
person_imgs = detector.crop_person(img, persons)

# extract person feature
feat = extractor.extract(person_imgs[0])
```

#### Distributed inferring (under test)

```
bash tools/dist_infer.sh scripts/detect_person.py 8
```

### Place Feature Extractor

The place featurer extractor is a Places365/ImageNet pretrained ResNet50.

Here is an example about how to extact place features for a given image.

```python
import cv2
from movienet.tools import PlaceExtractor

# build models
weight_path = './model/resnet50_places365.pth'
extractor = PlaceExtractor(weight_path, gpu=0)

# extract place feature
img_path = './tests/data/still01.jpg'
img = cv2.imread(img_path)
feat = extractor.extract(img)
```

You can also check the following demo python script.

```sh
python demos/place_demo.py
```

#### Distributed inferring

You can run it according to a file list in a distributed way

```sh
python tools/extract_place_feat.py --listfile tests/data/imglist.txt --img_prefix tests/data/aud_wav --save_path tests/data/place_feat --num_worker 2
```

### Action Feature Extractor

One could use the action extractor to obtain per-instance action feature from a
short video clip. Until now, we provide spatial-temporal action detection model
based on Fast-RCNN NonLocal-I3D-50 pre-trained on AVA dataset.

We provide two kinds of usage, namely, a lite version single video action extractor
and an action feature extract pipeline that efficiently extract action features
from a large movie database.

(1) Lite action extractor

```python
import mmcv

# bboxes should be n * 4 numpy array, normalized to range [0, 1].
# it could be the bboxes detected from the center frame of the provided video.
bboxes = [[x1, y1, x2, y2], ...]
imgs = [mmcv.imread(img_path) for img_path in frame_file_list]
extractor = ActionExtractor()
result = extractor.extract(imgs, bboxes)
```

(2) Extract pipeline

We also provide script for extracting multiple movies at high efficiency.
Please refer to the [code](https://github.com/movienet/movienet-tools/blob/master/scripts/extract_action_feats.py) for help.

The process of this pipeline will take a movie database (either videos or
extracted frames) as input and output the per-shot action features for each movie.
Therefore, to extract action features from a large movie database, one should
first run shot detection and save the results.
Also, person tracklets within each shot are needed. If the users do not run
person detection first, one could choose to detect the bounding boxes in
this script.

### Audio Feature Extractor

The audio feature extractor is to apply stft on wav to get its features, using [librosa](https://librosa.org/librosa/v0.4.0/index.html) toolkits

```python
from movienet.tools import AudioExtractor

# build models
extractor = AudioExtractor()

# extract audio feature
wav_path = './tests/data/aud_wav/shot_0001.wav'
feat = extractor.extract(wav_path)

```

You can also check the following demo python script.

```
python demos/audio_demo.py
```

#### Distributed inferring

You can run it according to a file list in a distributed way

```
python tools/extract_audio_feat.py --listfile tests/data/audlist.txt --aud_prefix tests/data/aud_wav --save_path tests/data/aud_stft --num_worker 2
```
