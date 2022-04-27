import torch
import torch.nn as nn

from .layers import FCNN

class SimpleModel(nn.Module):
    '''
    Simple model using dictionaries
    '''
    def __init__(self, in_dim, out_dim, hidden_dim, act='tanh'):
        super(SimpleModel, self).__init__()
        self.net = FCNN(in_dim, out_dim, hidden_dim, act)

    def forward(self, input):
        x = input['x']
        out = self.net(x)
        return {'dx':out}
