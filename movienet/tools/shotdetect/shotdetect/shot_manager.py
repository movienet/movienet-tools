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
import math

import cv2

from .frame_timecode import FrameTimecode
from .platform import get_csv_writer, tqdm
from .stats_manager import FrameMetricRegistered


def get_shots_from_cuts(cut_list, base_timecode, num_frames, start_frame=0):
    """Returns a list of tuples of start/end FrameTimecodes for each shot based
    on a list of detected shot cuts/breaks.

    This function is called when using the :py:meth:`shotManager.get_shot_list`
    method.
    The shot list is generated from a cutting list
    noting that each shot is contiguous, starting from the first to last frame
    of the input.
    Arguments:
        cut_list (List[FrameTimecode]): List of FrameTimecode objects where
            shot cuts/breaks occur.
        base_timecode (FrameTimecode): The base_timecode of which all
            FrameTimecodes in the cut_list are based on.
        num_frames (int or FrameTimecode): The number of frames, or
            FrameTimecode representing duration, of the video that was
            processed
            (used to generate last shot's end time).
        start_frame (int or FrameTimecode): The start frame or FrameTimecode
            of the cut list.
            Used to generate the first shot's start time.
    Returns:
        List of tuples in the form (start_time, end_time), where both
        start_time and
        end_time are FrameTimecode objects representing the exact time/frame
        where each shot occupies based on the input cut_list.
    """
    # shot list, where shots are tuples of
    # (Start FrameTimecode, End FrameTimecode).
    shot_list = []
    if not cut_list:
        shot_list.append(
            (base_timecode + start_frame, base_timecode + num_frames))
        return shot_list
    # Initialize last_cut to the first frame we processed,as it will be
    # the start timecode for the first shot in the list.
    last_cut = base_timecode + start_frame
    for cut in cut_list:
        shot_list.append((last_cut, cut))
        last_cut = cut
    # Last shot is from last cut to end of video.
    shot_list.append((last_cut, base_timecode + num_frames))

    return shot_list


def write_shot_list(output_csv_file, shot_list, cut_list=None):
    """Writes the given list of shots to an output file handle in CSV format.

    Arguments:
        output_csv_file: Handle to open file in write mode.
        shot_list: List of pairs of FrameTimecodes denoting each shot's
            sstart/end FrameTimecode.
        cut_list: Optional list of FrameTimecode objects denoting the cut list
            (i.e. the frames
            in the video that need to be split to generate individual shots).
            If not passed,
            the start times of each shot (besides the 0th shot)
            is used instead.
    """
    csv_writer = get_csv_writer(output_csv_file)
    # Output Timecode List
    csv_writer.writerow(['Timecode List:'] + cut_list if cut_list else
                        [start.get_timecode() for start, _ in shot_list[1:]])
    csv_writer.writerow([
        'shot Number', 'Start Frame', 'Start Timecode', 'Start Time (seconds)',
        'End Frame', 'End Timecode', 'End Time (seconds)', 'Length (frames)',
        'Length (timecode)', 'Length (seconds)'
    ])
    for i, (start, end) in enumerate(shot_list):
        duration = end - start
        csv_writer.writerow([
            '%d' % (i + 1),
            '%d' % start.get_frames(),
            start.get_timecode(),
            '%.3f' % start.get_seconds(),
            '%d' % end.get_frames(),
            end.get_timecode(),
            '%.3f' % end.get_seconds(),
            '%d' % duration.get_frames(),
            duration.get_timecode(),
            '%.3f' % duration.get_seconds()
        ])


class ShotManager(object):
    """The ShotManager class facilitates detection of shots via the
    :py:meth:`detect_shots` method, given a video source
    (:py:class:`VideoManager <shotdetect.video_manager.VideoManager>` or
    cv2.VideoCapture), and shotDetector algorithms added via the
    :py:meth:`add_detector` method.

    Can also optionally take a StatsManager instance during construction
    to cache intermediate shot detection calculations, making subsequent
    calls to :py:meth:`detect_shots` much faster, allowing the cached
    values to be saved/loaded to/from disk, and also manually
    determining the optimal threshold values or other options for
    various detection algorithms.
    """

    def __init__(self, stats_manager=None):
        self._cutting_list = []
        self._detector_list = []
        self._stats_manager = stats_manager
        self._num_frames = 0
        self._start_frame = 0

    def add_detector(self, detector):
        """Adds/registers a shotDetector (e.g. ContentDetector,
        ThresholdDetector) to run when detect_shots is called.

        The shotManager owns the detector object,
        so a temporary may be passed.
        Arguments:
            detector (shotDetector): shot detector to add to the shotManager.
        """
        detector.stats_manager = self._stats_manager
        self._detector_list.append(detector)
        if self._stats_manager is not None:
            # Allow multiple detection algorithms of the same type to be added
            # by suppressing any FrameMetricRegistered exceptions due to
            # attempts to re-register the same frame metric keys.
            try:
                self._stats_manager.register_metrics(detector.get_metrics())
            except FrameMetricRegistered:
                pass

    def get_num_detectors(self):
        """Gets number of registered shot detectors added via add_detector."""
        return len(self._detector_list)

    def clear(self):
        """Clears all cuts/shots and resets the shotManager's position.

        Any statistics generated are still saved in the StatsManager
        object passed to the shotManager's constructor, and thus,
        subsequent calls to detect_shots, using the same frame source
        reset at the initial time (if it is a VideoManager, use the
        reset() method), will use the cached frame metrics that were
        computed and saved in the previous call to detect_shots.
        """
        self._cutting_list.clear()
        self._num_frames = 0
        self._start_frame = 0

    def clear_detectors(self):
        """Removes all shot detectors added to the shotManager via
        add_detector()."""
        self._detector_list.clear()

    def get_shot_list(self, base_timecode):
        """Returns a list of tuples of start/end FrameTimecodes for each shot.

        The shot list is generated by calling :py:func:`get_shots_from_cuts`
        on the cutting list from :py:meth:`get_cut_list`, noting that
        each shot is contiguous, starting from
        the first and ending at the last frame of the input.
        Returns:
            List of tuples in the form (start_time, end_time), where both
            start_time and
            end_time are FrameTimecode objects representing the
            exact time/frame where each
            detected shot in the video begins and ends.
        """
        return get_shots_from_cuts(
            self.get_cut_list(base_timecode), base_timecode, self._num_frames,
            self._start_frame)

    def get_cut_list(self, base_timecode):
        """Returns a list of FrameTimecodes of the detected shot changes/cuts.

        Unlike get_shot_list, the cutting list returns a list of
        FrameTimecodes representing the point in the input video(s)
        where a new shot was detected, and thus the frame
        where the input should be cut/split. The cutting list, in turn, is
        used to generate the shot list, noting that each shot is contiguous
        starting from the first frame and ending at the last frame detected.

        Returns:
            List of FrameTimecode objects denoting the points in time where a
            shot change was detected in the input video(s),
            which can also be passed to external tools
            for automated splitting of the input into individual shots.
        """

        return [
            FrameTimecode(cut, base_timecode)
            for cut in self._get_cutting_list()
        ]

    def _get_cutting_list(self):
        """Returns a sorted list of unique frame numbers of any detected shot
        cuts."""
        # Remove duplicates here by creating a set then back to a list
        # and sort it.
        return sorted(list(set(self._cutting_list)))

    def _add_cut(self, frame_num):
        # Adds a cut to the cutting list.
        self._cutting_list.append(frame_num)

    def _add_cuts(self, cut_list):
        # Adds a list of cuts to the cutting list.
        self._cutting_list += cut_list

    def _process_frame(self, frame_num, frame_im):
        """Adds any cuts detected with the current frame to the cutting
        list."""
        for detector in self._detector_list:
            self._add_cuts(detector.process_frame(frame_num, frame_im))

    def _is_processing_required(self, frame_num):
        """Is Processing Required: Returns True if frame metrics not in
        StatsManager, False otherwise."""
        return all([
            detector.is_processing_required(frame_num)
            for detector in self._detector_list
        ])

    def _post_process(self, frame_num):
        """Adds any remaining cuts to the cutting list after processing the
        last frame."""
        for detector in self._detector_list:
            self._add_cuts(detector.post_process(frame_num))

    def detect_shots(self,
                     frame_source,
                     end_time=None,
                     frame_skip=0,
                     show_progress=True):
        """Perform shot detection on the given frame_source using the added
        shotDetectors.

        Blocks until all frames in the frame_source have been processed.
        Results can be obtained by calling either the get_shot_list()
        or get_cut_list() methods.
        Arguments:
            frame_source (shotdetect.video_manager.VideoManager or
                cv2.VideoCapture):
                A source of frames to process (using frame_source.read() as in
                VideoCapture).
                VideoManager is preferred as it allows concatenation of
                multiple videos as well as seeking, by defining start time
                and end time/duration.
            end_time (int or FrameTimecode): Maximum number of frames to detect
                (set to None to detect all available frames). Only needed for
                OpenCV
                VideoCapture objects; for VideoManager objects, use
                set_duration() instead.
            frame_skip (int): Not recommended except for extremely high
                framerate videos.
                Number of frames to skip (i.e. process every 1 in N+1 frames,
                where N is frame_skip, processing only 1/N+1 percent of the
                video,
                speeding up the detection time at the expense of accuracy).
                `frame_skip` **must** be 0 (the default) when using a
                StatsManager.
            show_progress (bool): If True, and the ``tqdm`` module is
                available, displays
                a progress bar with the progress, framerate, and expected
                time to
                complete processing the video frame source.

        Returns:
            int: Number of frames read and processed from the frame source.

        Raises:
            ValueError: `frame_skip` **must** be 0 (the default)
                if the shotManager
                was constructed with a StatsManager object.
        """

        if frame_skip > 0 and self._stats_manager is not None:
            raise ValueError('frame_skip must be 0 when using a StatsManager.')

        start_frame = 0
        curr_frame = 0
        end_frame = None

        total_frames = math.trunc(frame_source.get(cv2.CAP_PROP_FRAME_COUNT))

        start_time = frame_source.get(cv2.CAP_PROP_POS_FRAMES)
        if isinstance(start_time, FrameTimecode):
            start_frame = start_time.get_frames()
        elif start_time is not None:
            start_frame = int(start_time)
        self._start_frame = start_frame

        curr_frame = start_frame

        if isinstance(end_time, FrameTimecode):
            end_frame = end_time.get_frames()
        elif end_time is not None:
            end_frame = int(end_time)

        if end_frame is not None:
            total_frames = end_frame

        if start_frame is not None and not isinstance(start_time,
                                                      FrameTimecode):
            total_frames -= start_frame

        if total_frames < 0:
            total_frames = 0

        progress_bar = None
        if tqdm and show_progress:
            progress_bar = tqdm(total=total_frames, unit='frames')
        try:

            while True:
                if end_frame is not None and curr_frame >= end_frame:
                    break
                # We don't compensate for frame_skip here as the
                # frame_skip option is not allowed when using a
                # StatsManager - thus, processing is
                # *always* required for *all* frames when frame_skip > 0.
                if (self._is_processing_required(self._num_frames +
                                                 start_frame)
                        or self._is_processing_required(self._num_frames +
                                                        start_frame + 1)):
                    ret_val, frame_im = frame_source.read()
                else:
                    ret_val = frame_source.grab()
                    frame_im = None
                if not ret_val:
                    break
                self._process_frame(self._num_frames + start_frame, frame_im)

                curr_frame += 1
                self._num_frames += 1
                if progress_bar:
                    progress_bar.update(1)

                if frame_skip > 0:
                    for _ in range(frame_skip):
                        if not frame_source.grab():
                            break
                        curr_frame += 1
                        self._num_frames += 1
                        if progress_bar:
                            progress_bar.update(1)

            self._post_process(curr_frame)

            num_frames = curr_frame - start_frame
        finally:

            if progress_bar:
                progress_bar.close()

        return num_frames
