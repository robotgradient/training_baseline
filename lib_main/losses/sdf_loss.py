import torch
import torch.nn as nn



class SDFLoss():
    def __init__(self, field='sdf', delta = 0.6):
        self.field = field
        self.delta = delta

    def loss_fn(self, model, model_input, ground_truth, val=False):
        loss_dict = dict()
        label = ground_truth[self.field].squeeze()

        ## Compute model output ##
        model_outputs = model(model_input, field = self.field)

        ## Reconstruction Loss ##
        loss = nn.L1Loss()
        pred_clip_sdf = torch.clip(model_outputs[self.field], 0, self.delta)
        target_clip_sdf = torch.clip(label, 0, self.delta)
        l_rec = loss(pred_clip_sdf, target_clip_sdf)

        ## Gradient Reconstruction Loss (Focus only in the direction of the vector and not the norm) ##




        ## Total Loss
        loss_dict[self.field] = l_rec

        info = {'sdf': model_outputs}
        return loss_dict, info
