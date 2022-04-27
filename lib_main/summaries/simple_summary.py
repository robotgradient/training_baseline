import matplotlib.pyplot as plt

from lib_main.visualization import compute_vector_field
from .utils import plot_to_tensorboard

def simple_summary(model, model_input, ground_truth, info, writer, iter, prefix="", device='cpu'):
    """Writes tensorboard summaries using tensorboardx api.

    :param writer: tensorboardx writer object.
    :param predictions: Output of forward pass.
    :param ground_truth: Ground truth.
    :param iter: Iteration number.
    :param prefix: Every summary will be prefixed with this string.
    """
    x_input = model_input['x'].detach().cpu().numpy()

    x, y, dx, dy = compute_vector_field(model.net, device, min_max=[[-1,-1],[2.5,2.5]])

    fig = plt.figure()

    plt.streamplot(x,y,dx,dy)
    plt.scatter(x_input[:,0], x_input[:,1])

    writer.add_figure('vector field', fig, iter)

