import numpy as np
import os
import torch
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt



class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, trajs, device, steps=20):
        'Initialization'
        dim = trajs[0].shape[1]

        self.x = []
        self.x_n = np.zeros((0, dim))
        for i in range(steps):
            tr_i_all = np.zeros((0,dim))
            for tr_i in  trajs:
                _trj = tr_i[i:i-steps,:]
                tr_i_all = np.concatenate((tr_i_all, _trj), 0)
                self.x_n = np.concatenate((self.x_n, tr_i[-1:,:]),0)
            self.x.append(tr_i_all)

        self.x = torch.from_numpy(np.array(self.x)).float().to(device)
        self.x_n = torch.from_numpy(np.array(self.x_n)).float().to(device)

        self.len_n = self.x_n.shape[0]
        self.len = self.x.shape[1]
        self.steps_length = steps
        self.step = steps - 1

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def set_step(self, step=None):
        if step is None:
            self.step = np.random.randint(1, self.steps_length-1)

    def __getitem__(self, index):
        'Generates one sample of data'

        X = self.x[0, index, :]
        X_1 = self.x[self.step, index, :]

        index = np.random.randint(self.len_n)
        X_N = self.x_n[index, :]

        return X, [X_1, int(self.step), X_N, index]


class SimpleDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, x, y, device):
        'Initialization'
        self.dim = x.shape[-1]

        self.x = torch.from_numpy(x).float().to(device)
        self.y = torch.from_numpy(y).float().to(device)

        self.len = self.x.shape[0]

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def __getitem__(self, index):
        'Generates one sample of data'

        X = self.x[index, :]
        Y = self.y[index, :]

        return X, Y


class CycleDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, trajs, device, trajs_phase, steps=20):
        'Initialization'
        dim = trajs[0].shape[1]

        self.x = []
        self.x_n = np.zeros((0, dim))
        for i in range(steps):
            tr_i_all = np.zeros((0,dim))
            for tr_i in  trajs:
                _trj = tr_i[i:i-steps,:]
                tr_i_all = np.concatenate((tr_i_all, _trj), 0)
                self.x_n = np.concatenate((self.x_n, tr_i[-1:,:]),0)
            self.x.append(tr_i_all)

        self.x = torch.from_numpy(np.array(self.x)).float().to(device)
        self.x_n = torch.from_numpy(np.array(self.x_n)).float().to(device)

        ## Phase ordering ##
        trp_all = np.zeros((0))
        for trp_i in  trajs_phase:
            _trjp = trp_i[:-steps]
            trp_all = np.concatenate((trp_all, _trjp), 0)
        self.trp_all = torch.from_numpy(trp_all).float().to(device)

        self.len_n = self.x_n.shape[0]
        self.len = self.x.shape[1]
        self.steps_length = steps
        self.step = steps - 1

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def set_step(self, step=None):
        if step is None:
            self.step = np.random.randint(1, self.steps_length-1)

    def __getitem__(self, index):
        'Generates one sample of data'

        X = self.x[0, index, :]
        X_1 = self.x[self.step, index, :]
        phase = self.trp_all[index]

        return X, [X_1, int(self.step), phase]


class VDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, trajs, v_trj=None, dt=0.01):
        'Initialization'
        dim = trajs[0].shape[1]
        dt = dt

        self.x = []
        self.dx = []

        if v_trj is None:
            for tr_i in  trajs:
                num_pts = tr_i.shape[0]
                demo_smooth = np.zeros_like(tr_i)
                window_size = int(2 * (25. * num_pts / 150 // 2) + 1)
                for j in range(dim):
                    try:
                        if window_size>3:
                            poly = 3
                        else:
                            poly = window_size-1
                        demo_smooth[:, j] = savgol_filter(tr_i[:, j], window_size, poly)
                    except:
                        print('fail')

                demo_vel = np.diff(demo_smooth, axis=0) / dt
                self.x.append(demo_smooth[:-1,:])
                self.dx.append(demo_vel)
        else:
            self.x = trajs
            self.dx = v_trj

        self.x_data = np.zeros((0, dim))
        self.dx_data = np.zeros((0, dim))
        for i in range(len(self.x)):
            self.x_data = np.concatenate((self.x_data, self.x[i]),0)
            self.dx_data = np.concatenate((self.dx_data, self.dx[i]),0)

        self.x_data = torch.from_numpy(np.array(self.x_data)).float()
        self.dx_data = torch.from_numpy(np.array(self.dx_data)).float()

        self.len = self.x_data.shape[0]

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def __getitem__(self, index):
        'Generates one sample of data'
        x = self.x_data[index,:]
        dx = self.dx_data[index, :]
        return {'x':x}, {'dx':dx}
