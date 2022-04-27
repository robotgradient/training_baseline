import torch
import numpy as np
from lib_main.utils.utils import to_numpy, to_torch
import matplotlib.pyplot as plt


def compute_vector_field(policy, device, min_max = [[-1,-1],[1,1]], fig_number=1):

    min_x = min_max[0][0]
    max_x = min_max[1][0]
    min_y = min_max[0][1]
    max_y = min_max[1][1]

    n_sample = 100
    x = np.linspace(min_x, max_x, n_sample)
    y = np.linspace(min_y, max_y, n_sample)

    xy = np.meshgrid(x, y)
    h = np.concatenate(xy[0])
    v = np.concatenate(xy[1])
    hv = torch.Tensor(np.stack([h, v]).T).float()
    if device is not None:
        hv = hv.to(device)

    vel = policy(hv)
    vel = to_numpy(vel)
    vel = np.nan_to_num(vel)

    vel_x = np.reshape(vel[:, 0], (n_sample, n_sample))
    vel_y = np.reshape(vel[:, 1], (n_sample, n_sample))
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
    speed = speed/(np.max(speed)+0.01)

    return xy[0], xy[1], vel_x, vel_y
