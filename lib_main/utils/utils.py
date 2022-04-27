import torch
import collections
import numpy as np
import os


def dict_to_device(ob, device):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_device(v, device) for k, v in ob.items()}
    else:
        return ob.to(device)

def to_numpy(x):
    return x.detach().cpu().numpy()


def to_torch(x, device):
    return torch.from_numpy(x).float().to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
