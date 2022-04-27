import numpy as np
import matplotlib.pyplot as plt

def plot_to_tensorboard(writer, fig, step, name):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    # img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8

    # Add figure in numpy "image" to TensorBoard writer
    writer.add_image('confusion_matrix', img, step)
    plt.close(fig)


def point_cloud(writer, iter, name, points_xyz, colors=None):
    point_size_config = {
        'material': {
            'cls': 'PointsMaterial',
            'size': 0.05
        }
    }

    if colors is None:
       colors = np.zeros_like(points_xyz)

    writer.add_mesh(name, vertices=points_xyz, colors=colors,
                     config_dict={"material": point_size_config}, global_step=iter)

