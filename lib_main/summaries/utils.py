import numpy as np



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

