import torch


class InfoNCE():

    def __init__(self, negative_sampler, field='ene'):
        self.field = field
        self.negative_sampling = negative_sampler

    def loss_fn(self, model, model_input, ground_truth, val=False):

        model_inputs, model_outputs, labels = self.compute_energy(model, model_input, val)

        loss_dict = dict()
        label = labels[self.field].squeeze()

        #loss_dict['energy'] = -1 * ((1-label) * torch.log(model_outputs['energy'] + 1e-5) + (label) * torch.log(1 - model_outputs['energy'] + 1e-5)).mean()
        loss_dict[self.field] = -1 * (label * torch.log(model_outputs[self.field] + 1e-5) + (1 - label) * torch.log(1 - model_outputs[self.field] + 1e-5)).mean()

        info = {'model_inputs':model_inputs, 'labels':label, 'model_outputs':model_outputs}

        return loss_dict, {self.field:info}

    def compute_energy(self, model, model_input, val=False):
        p = model_input['point_cloud']

        name = 'x_'+self.field

        x_pos = model_input[name]
        dim = x_pos.dim()
        ## Dim is 3 if we are using only 3D points, Dim is 4 if we are using Homogeneous transforms 4x4
        if dim==3:
            x_neg = self.negative_sampling.sample(model=model, batch_size = x_pos.shape[:-1])
            label_pos = torch.ones_like(x_pos[..., 0])
            label_neg = torch.zeros_like(x_neg[..., 0])
        else:
            x_neg = self.negative_sampling.sample(model=model, batch_size = x_pos.shape[:-2])
            label_pos = torch.ones_like(x_pos[..., 0, 0])
            label_neg = torch.zeros_like(x_neg[..., 0, 0])

        x_in  = torch.cat((x_pos, x_neg), 1)
        labels = torch.cat((label_pos, label_neg),-1)

        model_input[name] = x_in
        model_outputs = model(model_input, field='ene')

        return model_input, model_outputs, {self.field:labels}
