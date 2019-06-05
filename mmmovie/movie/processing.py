import os
import os.path as osp
import subprocess
import tempfile

import mmcv.video as mmv
from mmcv.utils import requires_executable


@requires_executable('ffmpeg')
def convert_movie(in_file, out_file, print_cmd=False, pre_options='',
                  **kwargs):
    """Convert a movie with ffmpeg.

    This provides a general api to ffmpeg, the executed command is::

        `ffmpeg -y <pre_options> -i <in_file> <options> <out_file>`

    Options(kwargs) are mapped to ffmpeg commands with the following rules:

    - key=val: "-key val"
    - key=True: "-key"
    - key=False: ""

    Args:
        in_file (str): Input video filename.
        out_file (str): Output video filename.
        pre_options (str): Options appears before "-i <in_file>".
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    mmv.convert_video(in_file, out_file, print_cmd, pre_options, **kwargs)


@requires_executable('ffmpeg')
def resize_movie(in_file,
                 out_file,
                 size=None,
                 ratio=None,
                 keep_ar=False,
                 log_level='info',
                 print_cmd=False,
                 **kwargs):
    """Resize a movie.

    Args:
        in_file (str): Input video filename.
        out_file (str): Output video filename.
        size (tuple or str): Expected size (w, h), eg, (320, 240) or (320, -1).
            can also be string: '1080P', '720P', '480P', '360P', '240P'
        ratio (tuple or float): Expected resize ratio, (2, 0.5) means
            (w*2, h*0.5).
        keep_ar (bool): Whether to keep original aspect ratio.
        log_level (str): Logging level of ffmpeg.
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    res_map = {
        '1080P': (1920, 1080),
        '720P': (1280, 720),
        '480P': (858, 480),
        '360P': (480, 360),
        '240P': (352, 240)
    }
    if size is not None:
        if isinstance(size, str):
            if size not in ['1080P', '720P', '480P', '360P', '240P']:
                raise ValueError('No such resolution: {}'.format(ratio))
            if keep_ar:
                size = (-2, res_map[size][1])
                keep_ar = False
            else:
                size = res_map[size]
    mmv.resize_video(in_file, out_file, size, ratio, keep_ar, log_level, print_cmd, **kwargs)


@requires_executable('ffmpeg')
def concat_movie(video_list,
                 out_file,
                 vcodec=None,
                 acodec=None,
                 log_level='info',
                 print_cmd=False,
                 **kwargs):
    """Concatenate multiple movies into a single one.

    Args:
        video_list (list): A list of video filenames
        out_file (str): Output video filename
        vcodec (None or str): Output video codec, None for unchanged
        acodec (None or str): Output audio codec, None for unchanged
        log_level (str): Logging level of ffmpeg.
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    mmv.concat_video(video_list, out_file, vcodec, acodec, log_level, print_cmd, **kwargs)


@requires_executable('ffmpeg')
def cut_movie_by_time(in_file,
                      out_file,
                      start=None,
                      end=None,
                      vcodec=None,
                      acodec=None,
                      log_level='info',
                      print_cmd=False,
                      **kwargs):
    """Cut a clip from a video with start and end time

    Args:
        in_file (str): Input video filename.
        out_file (str): Output video filename.
        start (None or float): Start time (in seconds).
        end (None or float): End time (in seconds).
        vcodec (None or str): Output video codec, None for unchanged.
        acodec (None or str): Output audio codec, None for unchanged.
        log_level (str): Logging level of ffmpeg.
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    mmv.cut_video(in_file, out_file, start, end, vcodec, acodec, log_level, print_cmd, **kwargs)


@requires_executable('ffmpeg')
def cut_movie_by_frame(in_file,
                      out_file,
                      start=None,
                      end=None,
                      vcodec=None,
                      acodec=None,
                      log_level='info',
                      print_cmd=False,
                      **kwargs):
    """Cut a clip from a video with start and end frame

    Args:
        in_file (str): Input video filename.
        out_file (str): Output video filename.
        start (None or int): Start frame.
        end (None or int): End frame.
        vcodec (None or str): Output video codec, None for unchanged.
        acodec (None or str): Output audio codec, None for unchanged.
        log_level (str): Logging level of ffmpeg.
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    raise NotImplementedError


@requires_executable('ffmpeg')
def extract_audio_stream(in_file,
                         out_file,
                         acodec=None,
                         log_level='info',
                         print_cmd=False,
                         **kwargs):
    """extract audio stream of a movie

    Args:
        in_file (str): Input video filename.
        out_file (str): Output audio filename.
        acodec (None or str): Output audio codec, None for unchanged.
        log_level (str): Logging level of ffmpeg.
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    raise NotImplementedError


@requires_executable('ffmpeg')
def extract_video_stream(in_file,
                         out_file,
                         size=None,
                         ratio=None,
                         keep_ar=False,
                         vcodec=None,
                         log_level='info',
                         print_cmd=False,
                         **kwargs):
    """extract audio stream of a movie

    Args:
        in_file (str): Input video filename.
        out_file (str): Output audio filename.
        size (tuple or str): Expected size (w, h), eg, (320, 240) or (320, -1).
            can also be string: '1080P', '720P', '480P', '360P', '240P'
        ratio (tuple or float): Expected resize ratio, (2, 0.5) means
            (w*2, h*0.5).
        keep_ar (bool): Whether to keep original aspect ratio.
        vcodec (None or str): Output video codec, None for unchanged.
        log_level (str): Logging level of ffmpeg.
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    raise NotImplementedError