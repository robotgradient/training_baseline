import matplotlib.pyplot as plt
import torch


def uniform_sample(dim=1, batch=1000, xmin=torch.zeros(1), xmax=torch.ones(1)):
    x = torch.rand(batch,dim)
    scale = xmax - xmin
    x = x*scale[None,:] + xmin[None,:]
    return x

def batch_rejection_sampling(function, num_samples, dim=1, device='cpu',
                             xmin=torch.zeros(1), xmax=torch.ones(1), ymax=torch.ones(1), batch=1000):
    samples = torch.zeros(0, dim).to(device)
    energies = torch.zeros(0).to(device)
    while samples.shape[0] < num_samples:
        x = uniform_sample(dim, batch, xmin=xmin, xmax=xmax).to(device)
        y = uniform_sample(batch=batch, xmax=ymax).to(device)
        ## Concatenate the points ##
        energy = function(x)
        samples  = torch.cat((samples, x[y.squeeze() < energy,:]))
        energies = torch.cat((energies, energy[y.squeeze() < energy]))
    return samples[:num_samples,:], energies[:num_samples]


if __name__ == '__main__':

    def model(x):
        return -x.pow(2).sum(-1).pow(.5) + 3.

    x = batch_rejection_sampling(model, dim=2, num_samples=1000000,
                                 xmin=-torch.ones(2), xmax=torch.ones(2), ymax=3*torch.ones(1))

    x = x.detach().cpu().numpy()
    plt.hist2d(x[:,0], x[:,1], bins=20)
    plt.show()
    print(x.shape())