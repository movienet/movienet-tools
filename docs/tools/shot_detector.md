### Shot Detector

To initialize a shot detector and to specify a source video path and an output path are all we need to detect shots in a video.

Below is an example.

```python
from movienet.tools import ShotDetector

# build shot detectors
# if desire to specify detected duration,
# set being/edn_frame/time, otherwise just leave it blank
sdt = ShotDetector(
    print_list=True,  # print results on command line
    save_keyf_txt=True,  # save key frame text list
    save_keyf=False,  # save key frame image
    split_video=False,  # save split video
    begin_frame=0,
    end_frame=2000)

# specify source video path and output path
video_path = osp.join('tests', 'data/test1.mp4')
out_dir = osp.join('tests', 'data')

# detect shot and save results
sdt.shotdetect(video_path, out_dir)
```

Or you can check the following demo python script, and detect shot, extract keyframe, generate shot video.

```sh
python demos/detect_shots.py
```

#### Distributed inferring (under test)

It is easy to utilize multiprocessing package. You may also check [here](https://github.com/AnyiRao/SceneSeg/blob/master/pre/ShotDetect/shotdetect_p.py) for a detailed reference code.

```python
import multiprocessing
from movienet.tools import ShotDetector

list_file = 'tests/data/videolist.txt'
source_path = 'tests/data'
save_data_root_path = 'tests/data'

# build shot detectors
sdt = ShotDetector(
    print_list=True,  # print results on command line
    save_keyf_txt=True,  # save key frame text list
    save_keyf=True,  # save key frame image
    split_video=True  # save split video
    )

# define a running function here for parallel process
def run(video_path, save_data_root_path):
    sdt.shotdetect(video_path, save_data_root_path)

# read the file list and multi-process it
video_list = [x.strip() for x in open(list_file)]  # the list of videos to be processed
pool = multiprocessing.Pool(processes=2)  # specify number of workers here
for video_id in video_list:
    video_path = osp.abspath(osp.join(args.source_path, video_id))
    pool.apply_async(run, args=(video_path, save_data_root_path))
pool.close()
pool.join()
```

#### Extract wav file for each shot video according to a list

Extract wav file for each video.
This is for extracting each shot's audio feature, since the input of audio feature extractor is a wav file.

```sh
python scripts/prepare_shot_wav.py --listfile tests/data/videolist.txt --src_video_path tests/data --save_path tests/data/aud_wav --n
```

The shot detector is modified from [PySceneDetect](https://pyscenedetect.readthedocs.io/en/latest/), which aims to detect the shot boundaries in a video.
