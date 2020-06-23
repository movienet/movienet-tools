import mmcv

__all__ = ['read_movie_list']


def read_movie_list(path):
    """ read movie list from txt file or json.
    """

    if path.endswith('txt'):
        return mmcv.list_from_file(path)
    elif path.endswith('json'):
        return mmcv.load(path)
    else:
        raise ValueError('File must be `txt` or `json` file.')
