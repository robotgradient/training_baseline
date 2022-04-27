import torch


def occupancy_loss(model_input, model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict

class OccupancyLoss():
    def __init__(self, field='occ'):
        self.field = field

    def loss_fn(self, model, model_input, ground_truth, val=False):
        loss_dict = dict()
        label = ground_truth[self.field].squeeze()

        ## Compute model output ##
        model_outputs = model(model_input, field = self.field)

        loss_dict[self.field] = -1 * (label * torch.log(model_outputs[self.field] + 1e-5) + (1 - label) * torch.log(
            1 - model_outputs[self.field] + 1e-5)).mean()

        info = {'occ': model_outputs}

        return loss_dict, info
