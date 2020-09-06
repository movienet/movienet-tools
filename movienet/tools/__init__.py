from .crawler import DoubanCrawler, IMDBCrawler, TMDBCrawler
from .movie import (MovieReader, concat_movie, convert_movie,
                    cut_movie_by_time, frames_to_seconds, resize_movie,
                    seconds_to_frames, seconds_to_timecode,
                    timecode_to_seconds)
from .shotdetect import ShotDetector
from .version import __version__, short_version

try:
    import torch
except ImportError:
    __all__ = [
        '__version__', 'short_version', 'DoubanCrawler', 'IMDBCrawler',
        'TMDBCrawler', 'MovieReader', 'ShotDetector', 'convert_movie',
        'resize_movie', 'cut_movie_by_time', 'concat_movie',
        'seconds_to_timecode', 'seconds_to_frames', 'frames_to_seconds',
        'timecode_to_seconds'
    ]
else:
    from .detector import DistPersonDetector, FaceDetector, PersonDetector
    from .extractor import (AudioExtractor, DistAudioExtractor,
                            DistPersonExtractor, DistPlaceExtractor,
                            FaceExtractor, PersonExtractor, PlaceExtractor)
    from .action_extractor import ActionExtractor
    from .metaio import MetaParser
    __all__ = [
        '__version__', 'short_version', 'DoubanCrawler', 'IMDBCrawler',
        'TMDBCrawler', 'MetaParser', 'MovieReader', 'ShotDetector',
        'convert_movie', 'resize_movie', 'cut_movie_by_time', 'concat_movie',
        'seconds_to_timecode', 'seconds_to_frames', 'frames_to_seconds',
        'timecode_to_seconds', 'PersonDetector', 'DistPersonDetector',
        'PlaceExtractor', 'DistPlaceExtractor', 'AudioExtractor',
        'DistAudioExtractor', 'PersonExtractor', 'DistPersonExtractor',
        'FaceDetector', 'FaceExtractor', 'ActionExtractor'
    ]
