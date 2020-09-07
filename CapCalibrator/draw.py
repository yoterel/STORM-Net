from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np


class Arrow3D(FancyArrowPatch):
    """
    draws a 3D arrow onto a renderer
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_3d_pc(ax, data, selected):
    """
    plots a 3d point cloud representation of data (nx3)
    :param ax:
    :param data:
    :return:
    """
    colors = ['b'] * len(data)
    colors[selected] = 'r'
    data_min = np.min(data, axis=0)
    a = Arrow3D([data_min[0], data_min[0]+3], [data_min[1], data_min[1]],
                [data_min[2], data_min[2]], mutation_scale=10,
                lw=1, arrowstyle="-|>", color="r")
    b = Arrow3D([data_min[0], data_min[0]], [data_min[1], data_min[1]+3],
                [data_min[2], data_min[2]], mutation_scale=10,
                lw=1, arrowstyle="-|>", color="r")
    c = Arrow3D([data_min[0], data_min[0]], [data_min[1], data_min[1]],
                [data_min[2], data_min[2]+3], mutation_scale=10,
                lw=1, arrowstyle="-|>", color="r")
    if selected < len(data) -1:
        d = Arrow3D([data[selected, 0], data[selected+1, 0]], [data[selected, 1], data[selected+1, 1]],
                    [data[selected, 2], data[selected+1, 2]], mutation_scale=10,
                    lw=1, arrowstyle="-|>", color="r")
        ax.add_artist(d)
    ax.add_artist(a)
    ax.add_artist(b)
    ax.add_artist(c)
    for i, (c, x, y, z) in enumerate(zip(colors, data[:, 0], data[:, 1], data[:, 2])):
        ax.scatter(x, y, z, marker='o', c=c)
        ax.text(x + 0.2, y + 0.2, z + 0.2, '%s' % (str(i)), size=6, zorder=1, color='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title('Point {} (WASD: change view, Arrows: next/previous point)'.format(selected))
