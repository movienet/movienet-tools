from torch.utils.data.sampler import Sampler
import numpy as np
import torch

class PaddedSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1, ngpu=1):
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.ngpu = ngpu
        nbatch = samples_per_gpu*ngpu
        self.num_samples = int(np.ceil(len(dataset) / nbatch)) * nbatch

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        num_extra = self.num_samples - len(indices)
        indices = np.concatenate([indices, indices[:num_extra]])
        indices = torch.from_numpy(indices).long()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples
