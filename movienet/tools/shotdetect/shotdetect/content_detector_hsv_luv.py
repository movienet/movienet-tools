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

import cv2
import numpy as np

from .shot_detector import shotDetector


class ContentDetectorHSVLUV(shotDetector):
    """Detects fast cuts using changes in colour and intensity between frames.

    Detect shot boundary using HSV and LUV information.
    """

    def __init__(self, threshold=30.0, min_shot_len=15):
        super(ContentDetectorHSVLUV, self).__init__()
        self.hsv_threshold = threshold
        self.delta_hsv_gap_threshold = 10
        self.luv_threshold = 40
        self.hsv_weight = 5
        # minimum length (frames length) of any given shot
        self.min_shot_len = min_shot_len
        self.last_frame = None
        self.last_shot_cut = None
        self.last_hsv = None
        self._metric_keys = [
            'hsv_content_val', 'delta_hsv_hue', 'delta_hsv_sat',
            'delta_hsv_lum', 'luv_content_val', 'delta_luv_hue',
            'delta_luv_sat', 'delta_luv_lum'
        ]
        self.cli_name = 'detect-content'
        self.last_luv = None

    def process_frame(self, frame_num, frame_img):
        """Similar to ThresholdDetector, but using the HSV colour space
        DIFFERENCE instead of single-frame RGB/grayscale intensity (thus cannot
        detect slow fades with this method).

        Args:
            frame_num (int): Frame number of frame that is being passed.

            frame_img (Optional[int]): Decoded frame image (np.ndarray) to
                perform shot detection on. Can be None *only* if the
                self.is_processing_required() method
                (inhereted from the base shotDetector class) returns True.

        Returns:
            List[int]: List of frames where shot cuts have been detected.
            There may be 0 or more frames in the list, and not necessarily
            the same as frame_num.
        """
        cut_list = []
        metric_keys = self._metric_keys
        _unused = ''

        if self.last_frame is not None:
            # Change in average of HSV (hsv), (h)ue only,
            # (s)aturation only, (l)uminance only.
            delta_hsv_avg, delta_hsv_h, delta_hsv_s, delta_hsv_v = \
                0.0, 0.0, 0.0, 0.0
            delta_luv_avg, delta_luv_h, delta_luv_s, delta_luv_v = \
                0.0, 0.0, 0.0, 0.0

            if (self.stats_manager is not None
                    and self.stats_manager.metrics_exist(
                        frame_num, metric_keys)):
                delta_hsv_avg, delta_hsv_h, delta_hsv_s, delta_hsv_v, \
                    delta_luv_avg, delta_luv_h, delta_luv_s, delta_luv_v = \
                    self.stats_manager.get_metrics(
                        frame_num, metric_keys)

            else:
                num_pixels = frame_img.shape[0] * frame_img.shape[1]
                curr_luv = cv2.split(
                    cv2.cvtColor(frame_img, cv2.COLOR_BGR2Luv))
                curr_hsv = cv2.split(
                    cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))
                last_hsv = self.last_hsv
                last_luv = self.last_luv
                if not last_hsv:
                    last_hsv = cv2.split(
                        cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2HSV))
                    last_luv = cv2.split(
                        cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2Luv))

                delta_hsv = [0, 0, 0, 0]
                for i in range(3):
                    num_pixels = curr_hsv[i].shape[0] * curr_hsv[i].shape[1]
                    curr_hsv[i] = curr_hsv[i].astype(np.int32)
                    last_hsv[i] = last_hsv[i].astype(np.int32)
                    delta_hsv[i] = np.sum(
                        np.abs(curr_hsv[i] - last_hsv[i])) / float(num_pixels)
                delta_hsv[3] = sum(delta_hsv[0:3]) / 3.0
                delta_hsv_h, delta_hsv_s, delta_hsv_v, delta_hsv_avg = \
                    delta_hsv

                delta_luv = [0, 0, 0, 0]
                for i in range(3):
                    num_pixels = curr_luv[i].shape[0] * curr_luv[i].shape[1]
                    curr_luv[i] = curr_luv[i].astype(np.int32)
                    last_luv[i] = last_luv[i].astype(np.int32)
                    delta_luv[i] = np.sum(
                        np.abs(curr_luv[i] - last_luv[i])) / float(num_pixels)
                delta_luv[3] = sum(delta_luv[0:3]) / 3.0
                delta_luv_h, delta_luv_s, delta_luv_v, delta_luv_avg = \
                    delta_luv

                if self.stats_manager is not None:
                    self.stats_manager.set_metrics(
                        frame_num, {
                            metric_keys[0]: delta_hsv_avg,
                            metric_keys[1]: delta_hsv_h,
                            metric_keys[2]: delta_hsv_s,
                            metric_keys[3]: delta_hsv_v,
                            metric_keys[0 + 4]: delta_luv_avg,
                            metric_keys[1 + 4]: delta_luv_h,
                            metric_keys[2 + 4]: delta_luv_s,
                            metric_keys[3 + 4]: delta_luv_v,
                        })

                self.last_hsv = curr_hsv
                self.last_luv = curr_luv
            if delta_hsv_avg >= self.hsv_threshold and \
                    delta_hsv_avg - self.hsv_threshold >= \
                    self.delta_hsv_gap_threshold:
                if self.last_shot_cut is None \
                    or (
                        (frame_num - self.last_shot_cut) >= self.min_shot_len
                        ):
                    cut_list.append(frame_num)
                    self.last_shot_cut = frame_num
            elif delta_hsv_avg >= self.hsv_threshold and \
                    delta_hsv_avg - self.hsv_threshold < \
                    self.delta_hsv_gap_threshold and \
                    delta_luv_avg + self.hsv_weight * \
                    (delta_hsv_avg - self.hsv_threshold) > self.luv_threshold:
                if self.last_shot_cut is None \
                    or (
                        (frame_num - self.last_shot_cut) >= self.min_shot_len
                        ):
                    cut_list.append(frame_num)
                    self.last_shot_cut = frame_num

            if self.last_frame is not None and self.last_frame is not _unused:
                del self.last_frame

        # If we have the next frame computed, don't copy the current frame
        # into last_frame since we won't use it on the next call anyways.
        if (self.stats_manager is not None and
                self.stats_manager.metrics_exist(frame_num + 1, metric_keys)):
            self.last_frame = _unused
        else:
            self.last_frame = frame_img.copy()
        return cut_list
