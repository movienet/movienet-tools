# The codes below partially refer to the PySceneDetect. According
# to its BSD 3-Clause License, we keep the following.
#
#          PySceneDetect: Python-Based Video Scene Detector
#   ---------------------------------------------------------------
#     [  Site: http://www.bcastell.com/projects/PySceneDetect/   ]
#     [  Github: https://github.com/Breakthrough/PySceneDetect/  ]
#     [  Documentation: http://pyscenedetect.readthedocs.org/    ]
#
# Copyright (C) 2014-2020 Brandon Castellano <http://www.bcastell.com>.
#
# PySceneDetect is licensed under the BSD 3-Clause License; see the included
# LICENSE file, or visit one of the following pages for details:
#  - https://github.com/Breakthrough/PySceneDetect/
#  - http://www.bcastell.com/projects/PySceneDetect/
#
# This software uses Numpy, OpenCV, click, tqdm, simpletable, and pytest.
# See the included LICENSE files or one of the above URLs for more information.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import print_function
import os
import os.path as osp

from .shotdetect import ContentDetectorHSVLUV
from .shotdetect.keyf_img_saver import generate_images, generate_images_txt
from .shotdetect.shot_manager import ShotManager
from .shotdetect.stats_manager import StatsManager
from .shotdetect.video_manager import VideoManager
from .shotdetect.video_splitter import split_video_ffmpeg


class ShotDetector(object):
    """Shot detector class with options. print_list (bool): Whether to print
    detect shot list out. keep_resolution (bool): Whether to keep resolution in
    the process of shot detect save_keyf (bool): Whether to save key frame
    image. save_keyf_txt (bool): Whether to save key frame image index.
    split_video (bool): Whether to split shot video out. begin_time/end_time
    (float): Set up duration time. begin_frame/end_frame (int): Set up duration
    frame.

    :Example:
    >>> from movienet.tools import ShotDetector
    >>> sdt = ShotDetector(print_list=True, begin_frame=0,end_frame=2000)
    >>> video_path = osp.join('sample.mp4')
    >>> out_dir = "./"
    >>> sdt.shotdetect(video_path,out_dir)
    """

    def __init__(self,
                 print_list=False,
                 keep_resolution=False,
                 save_keyf=False,
                 save_keyf_txt=False,
                 save_stat_csv=False,
                 split_video=False,
                 begin_time=None,
                 end_time=100.0,
                 begin_frame=None,
                 end_frame=1000):
        self.print_list = print_list
        self.keep_resolution = keep_resolution
        self.save_keyf = save_keyf
        self.save_keyf_txt = save_keyf_txt
        self.save_stat_csv = save_stat_csv
        self.split_video = split_video
        self.begin_time = begin_time
        self.end_time = end_time
        self.begin_frame = begin_frame
        self.end_frame = end_frame

    def shotdetect(self, video_path, out_dir, show_progress=True):
        """Detect shots from a video.

        Args:
            video_path (str): The path to the video.
            out_dir (str): Output directory to store all shot data including
            shot detect statistic, shot detect txt, key frame image, and shot
            video.
        """
        video_path = video_path
        stats_file_folder_path = osp.join(out_dir, 'shot_stats')
        os.makedirs(stats_file_folder_path, exist_ok=True)
        video_prefix = video_path.rsplit('.', 1)[0].rsplit('/', 1)[-1]
        stats_file_path = osp.join(stats_file_folder_path,
                                   '{}.csv'.format(video_prefix))

        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()
        shot_manager = ShotManager(stats_manager)

        # Add ContentDetector algorithm
        shot_manager.add_detector(ContentDetectorHSVLUV())
        base_timecode = video_manager.get_base_timecode()
        shot_list = []
        try:
            # If stats file exists, load it.
            if osp.exists(stats_file_path):
                # Read stats from CSV file opened in read mode:
                with open(stats_file_path, 'r') as stats_file:
                    stats_manager.load_from_csv(stats_file, base_timecode)

            # Set begin and end time
            if self.begin_time is not None:
                start_time = base_timecode + self.begin_time
                end_time = base_timecode + self.end_time
                video_manager.set_duration(
                    start_time=start_time, end_time=end_time)
            elif self.begin_frame is not None:
                start_frame = base_timecode + self.begin_frame
                end_frame = base_timecode + self.end_frame
                video_manager.set_duration(
                    start_time=start_frame, end_time=end_frame)
            if self.keep_resolution:
                video_manager.set_downscale_factor(1)

            else:
                # Automatically set resolution downscale factor to improve
                # processing speed.
                video_manager.set_downscale_factor()
            video_manager.start()
            # Perform shot detection on video_manager.
            shot_manager.detect_shots(
                frame_source=video_manager, show_progress=show_progress)
            # Obtain list of detected shots.
            shot_list = shot_manager.get_shot_list(base_timecode)
            # Each shot is a tuple of (start, end) FrameTimecodes.
            if self.print_list:
                print('List of shots obtained:')
                for i, shot in enumerate(shot_list):
                    print('Shot %4d: Start %s / Frame %d, End %s / Frame %d' %
                          (
                              i,
                              shot[0].get_timecode(),
                              shot[0].get_frames(),
                              shot[1].get_timecode(),
                              shot[1].get_frames(),
                          ))
            # Save keyf img for each shot
            if self.save_keyf:
                output_dir = osp.join(out_dir, 'shot_keyf', video_prefix)
                generate_images(video_manager, shot_list, output_dir)

            if self.save_keyf_txt:
                output_dir = osp.join(out_dir, 'shot_txt',
                                      '{}.txt'.format(video_prefix))
                os.makedirs(osp.join(out_dir, 'shot_txt'), exist_ok=True)
                generate_images_txt(shot_list, output_dir)

            # Split video into shot video
            if self.split_video:
                output_dir = osp.join(out_dir, 'shot_split_video',
                                      video_prefix)
                split_video_ffmpeg([video_path],
                                   shot_list,
                                   output_dir,
                                   suppress_output=True)

            # We only write to the stats file if a save is required:
            if self.save_stat_csv and stats_manager.is_save_required():
                with open(stats_file_path, 'w') as stats_file:
                    stats_manager.save_to_csv(stats_file, base_timecode)
        finally:
            video_manager.release()
