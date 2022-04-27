import torch
import torch.nn as nn
import torch.distributions as dist
from abc import ABC, abstractmethod
import ghalton
import numpy as np
import math
from pytorch3d.transforms import random_rotations

class SamplerBase(ABC):

    def __init__(self, sampler, dim, num_samples, device):

        self.sampler = sampler
        self.dim = dim
        self.device = device
        self.num_samples = num_samples

    def train(self, **kwargs):
        pass

    @abstractmethod
    def sample(self, **kwargs):
        pass

    def make_train(self):
        pass

    def make_eval(self):
        pass


class Uniform(SamplerBase):

    def __init__(
            self,
            lows: list,
            highs: list,
            dim: int = None,
            num_samples: int = None,
            device='cpu',
    ):
        assert len(lows) == dim
        assert len(highs) == dim

        self.lows = torch.tensor(lows).to(device)
        self.highs = torch.tensor(highs).to(device)
        sampler = dist.Uniform(self.lows, self.highs)

        super().__init__(sampler, dim, num_samples, device)

    def train(self, model, batch_size, context_input):
        pass

    def sample(self, model, batch_size, context_input=None):
        # return self.sampler.sample((batch_size, self.num_samples)).squeeze(0).to(self.device)
        return self.sampler.sample(batch_size).to(self.device)


class SE3_Uniform(SamplerBase):

    def __init__(
            self,
            t_lows: list,
            t_highs: list,
            num_samples: int = None,
            device='cpu',
    ):
        dim = 6
        self.lows = torch.tensor(t_lows).to(device)
        self.highs = torch.tensor(t_highs).to(device)
        sampler = dist.Uniform(self.lows, self.highs)

        super().__init__(sampler, dim, num_samples, device)

    def train(self, model, batch_size, context_input):
        pass


    def _sample_batch_rotations(self, batch_size):
        n = batch_size[0]*batch_size[1]
        rot = random_rotations(n)
        R = rot.reshape(batch_size[0], batch_size[1], 3, 3)
        return R

    def _build_batch_H(self, trans, rots):
        x = torch.eye(4)
        x = x.reshape((1, 1, 4, 4))
        H = x.repeat(trans.shape[0], trans.shape[1], 1, 1)
        H[:, :, :-1, -1] = trans
        H[:, :, :-1, :-1] = rots
        return H

    def sample(self, model, batch_size, context_input=None):
        trans = self.sampler.sample(batch_size)
        rots = self._sample_batch_rotations(batch_size)
        H = self._build_batch_H(trans, rots)
        return H.to(self.device)

