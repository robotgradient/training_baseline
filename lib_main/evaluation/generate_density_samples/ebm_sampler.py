import torch
from lib_main.utils import dict_to_device
from lib_main.evaluation.generate_density_samples import batch_rejection_sampling


class GeneratorEBM():
    def __init__(self, model, device='cpu', num_samples=1000, dim =3,
                 xmin=-1.5*torch.ones(3), xmax=1.5*torch.ones(3), ymax=torch.ones(1)):
        self.model = model
        self.device = device
        self.num_samples = num_samples
        self.dim = dim
        self.xmin = xmin
        self.xmax = xmax
        self.ymax = ymax

    def eval_points(self, x):
        input = {'x_ene': x[None, ...]}
        with torch.no_grad():
            ene_out = self.model.field_decoder(input, self.context, field='ene')
        return ene_out['ene'].squeeze()

    def generate_samples(self, data):
        ## Generate samples by rejection-sampling
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = dict_to_device(data, device)
        with torch.no_grad():
            self.context = self.model.point_encoder(inputs)

        samples, energies = batch_rejection_sampling(function=self.eval_points, dim= self.dim,
                                           num_samples=self.num_samples, device=self.device,
                                           xmin=self.xmin, xmax=self.xmax, ymax=self.ymax)
        return samples, energies