## Getting Started
### Table of Contents
1. [Shot Detector](#shot-detector)
2. [Featur Extractor](#featur-extractor)
3. [Crawler](#crawler)


### Shot Detector
##### Detect shot, extract keyframe, generate shot video
```
python demos/detect_shots.py
```

##### Prepare shot wav
```
python tools/prepare_shot_wav.py --listfile tests/data/videolist.txt --src_video_path tests/data --save_path tests/data/aud_wav --num_worker 2
```

### Featur Extractor

#### Audio 

##### Single
```
python demos/audio_demo.py
```

##### Distributed
```
python tools/extract_audio_feat.py --listfile tests/data/audlist.txt --aud_prefix tests/data/aud_wav --save_path tests/data/aud_stft --num_worker 2
```

### Crawler
