import torch
import torch.nn as nn

class L1Loss():
    def __init__(self, field='dx'):
        self.field = field

    def loss_fn(self, model, model_input, ground_truth, val=False):
        loss_dict = dict()

        ## Compute model output ##
        model_outputs = model(model_input)

        ## L1 Loss
        loss = nn.L1Loss()
        target = ground_truth[self.field].squeeze()
        prediction = model_outputs['dx']
        l1 = loss(prediction, target)

        loss_dict[self.field] = l1

        info = {self.field: model_outputs}

        return loss_dict, info
