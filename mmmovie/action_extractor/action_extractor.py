import mmcv
from detecotrs import FastRCNN


class ActionExtractor(object):

    def __init__(self, config_path, weight_path, gpu=0):
        cfg = mmcv.Config.fromfile(config_path)
        self.model = FastRCNN(**cfg)
        mmcv.runner.load_checkpoint(weight_path, map_location=f"cuda:{gpu}")
        self.data_preprocessor = data_preprocessor(gpu)
