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
    mmv.resize_video(in_file, out_file, size, ratio, keep_ar, log_level,
                     print_cmd, **kwargs)


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
    mmv.concat_video(video_list, out_file, vcodec, acodec, log_level,
                     print_cmd, **kwargs)


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
    mmv.cut_video(in_file, out_file, start, end, vcodec, acodec, log_level,
                  print_cmd, **kwargs)


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
                         audio_rate=None,
                         byte_rate=None,
                         log_level='info',
                         overwrite=False,
                         print_cmd=False,
                         **kwargs):
    """extract audio stream of a movie

    Args:
        in_file (str): Input video filename.
        out_file (str): Output audio filename.
        acodec (None or str): Output audio codec, None for unchanged.
        audio_rate (None or int): audio rate, e.g., 16000.
        byte_rate (None or str or int): byte rate, e.g., 128k
        log_level (str): Logging level of ffmpeg.
        overwrite (bool): whether to overwrite the file
            if out_file already exists.
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    assert log_level in [
        'quiet', 'panic', 'fatal', 'error', 'warning', 'info', 'verbose',
        'debug', 'trace'
    ]
    acodec_str = acodec if acodec is not None else 'copy'
    ar_str = 'ar {}'.format(audio_rate) if audio_rate is not None else ''
    br_str = '-b:a {}'.format(byte_rate) if byte_rate is not None else ''
    overwrite_str = '-y' if overwrite else '-n'
    cmd_tmpl = 'ffmpeg {} -loglevel {} -i {} -vn -c:a {} {} {} {}'
    cmd = cmd_tmpl.format(overwrite_str, log_level, in_file, acodec_str,
                          ar_str, br_str, out_file)
    ret = os.system(cmd)
    if ret == 2:
        print('Capture keyboard interrupt when execute cmd ``{}``.\nExit!'.
              format(cmd))
        exit()


@requires_executable('ffmpeg')
def extract_video_stream(in_file,
                         out_file,
                         size=None,
                         vcodec=None,
                         log_level='info',
                         print_cmd=False,
                         crf=None,
                         pix_fmt=None,
                         mute=True,
                         overwrite=False):
    """extract audio stream of a movie

    Args:
        in_file (str): Input video filename.
        out_file (str): Output audio filename.
        size (tuple): Expected size (w, h), eg, (320, 240) or (320, -2).
        vcodec (None or str): Output video codec, None for unchanged.
        log_level (str): Logging level of ffmpeg.
        print_cmd (bool): Whether to print the final ffmpeg command.
        crf (None or int): change crf value if specified.
        pix_fmt (None or str): pixel format, e.g., yuv420p.
        mute (bool): whether to extract audio stream.
        overwrite (bool): whether to overwrite the file
            if out_file already exists.

    """
    # TODO: add make_exists option.

    assert isinstance(size, tuple) and len(size) == 2
    assert log_level in [
        'quiet', 'panic', 'fatal', 'error', 'warning', 'info', 'verbose',
        'debug', 'trace'
    ]

    cmd_tmpl = 'ffmpeg {} -loglevel {} -i {} -vcodec {} {} {} {} {} {}'
    vcodec_str = vcodec if vcodec is not None else 'copy'
    scale_str = '-filter:v scale={}:{}'.format(
        size[0], size[1]) if size is not None else ''
    crf_str = '-crf {}'.format(crf) if crf is not None else ''
    pix_fmt_str = '-pix_fmt {}'.format(pix_fmt) if pix_fmt is not None else ''
    mute_str = '-an' if mute else ''
    overwrite_str = '-y' if overwrite else '-n'
    cmd = cmd_tmpl.format(overwrite_str, log_level, in_file, vcodec_str,
                          scale_str, crf_str, pix_fmt_str, mute_str, out_file)
    ret = os.system(cmd)
    if ret == 2:
        print('Capture keyboard interrupt when execute cmd ``{}``.\nExit!'.
              format(cmd))
        exit()
