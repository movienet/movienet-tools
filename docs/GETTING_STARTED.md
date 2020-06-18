## Getting Started

### Shot Detector
pass

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