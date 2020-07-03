from torchvision.transforms import Compose, ToTensor

from .transforms import Normalize, OneImageCollate


class BaseDataProcessor(object):

    def __init__(self, gpu=0):
        self.pipeline = self.build_data_pipline(gpu)

    def __call__(self, img):
        """process an image.

        Args:
            img (np.array<uint8>): the input image, in BGR
        """
        return self.pipeline(img)

    def build_data_pipline(self, gpu):
        raise NotImplementedError


class FaceDataProcessor(BaseDataProcessor):
    """image preprocess pipeline for place feature extractor."""

    def build_data_pipline(self, gpu):
        pipeline = Compose([
            Normalize(
                mean=[127.5, 127.5, 127.5],
                std=[128.0, 128.0, 128.0],
                to_rgb=True),
            ToTensor(),
            OneImageCollate(gpu)
        ])
        return pipeline
