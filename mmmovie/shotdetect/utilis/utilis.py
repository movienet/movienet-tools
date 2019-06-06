from collections import Counter
import numpy as np
import os
import pdb

def mkdir_ifmiss(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_folder_list(checked_directory,log_fn):
    checked_list = os.listdir(checked_directory)
    with open(log_fn, "w") as f:
        for item in checked_list:
            f.write(item+"\n")
def strcal(shotid,num):
    return str(int(shotid)+num).zfill(4)

def timecode_to_frames(timecode,framerate):
    return int(int(timecode.split(",")[1])*0.001*framerate) + sum(f * int(t) for f,t in zip((3600*framerate, 60*framerate, framerate), timecode.split(",")[0].split(':')))

def frames_to_timecode(frames,framerate):
    ms = "{0:.3f}".format((frames % framerate)/framerate).split(".")[1]
    return '{0:02d}:{1:02d}:{2:02d},{3:s}'.format(int(frames / (3600*framerate)),
                                            int(frames / (60*framerate) % 60),
                                            int(frames / framerate % 60),
                                            ms)

if __name__ == '__main__':
    framerate = int(24)
    print (timecode_to_frames('00:01:08,903',framerate))
    print (frames_to_timecode(1653,framerate))
    
    # frames = 1657 
    # ms = "{0:.3f}".format((frames % framerate)/framerate).split(".")[1]
