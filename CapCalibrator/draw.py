import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import cv2
from pathlib import Path
import utils
from file_io import load_db

image_hsv = None
pixel = (0, 0, 0) #RANDOM DEFAULT VALUE
ftypes = [
    ('JPG', '*.jpg;*.JPG;*.JPEG'),
    ('PNG', '*.png;*.PNG'),
    ('GIF', '*.gif;*.GIF'),
]


def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y, x]

        #HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
        upper = np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower = np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        print(lower, upper)

        #A MONOCHROME MASK FOR GETTING A BETTER VISION OVER THE COLORS
        image_mask = cv2.inRange(image_hsv, lower, upper)
        cv2.imshow("Mask", image_mask)


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


def plot_3d_pc(ax, data, selected, names=None):
    """
    plots a 3d point cloud representation of data
    :param ax: the axis to plot into
    :param data: the data (nx3 numpy array)
    :param selected: an int representing the currently selected data point - will be painted red
    :param names: the names of the data points
    :return:
    """
    if not names:
        names = [str(i) for i in range(len(data))]
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
        ax.text(x + 0.2, y + 0.2, z + 0.2, '%s' % (names[i]), size=6, zorder=1, color='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title('Point {} (WASD: change view, Arrows: next/previous point)'.format(selected))


def visualize_annotated_data(db_path, filter=None):
    my_format = "pickle" if db_path.suffix == ".pickle" else "json"
    db = load_db(db_path, my_format, filter)
    # db = fix_db(db)
    # save_db(db, db_path)
    shift = 0
    for key in db.keys():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data = db[key][shift]["data"][0]
        if len(db[key][shift]["data"][0].shape) < 3:
            data = np.expand_dims(data, axis=0)
        data[:, :, 0::2] /= 960
        data[:, :, 1::2] /= 540
        utils.center_data(data)
        data = np.squeeze(data)
        # s_linear = [n for n in range(len(data))]
        c = ['b', 'b', 'b', 'r', 'r', 'r', 'g']
        for t in range(0, data.shape[1], 2):
            x = data[:, t]
            y = data[:, t+1]
            exist = np.nonzero(x)
            x = x[exist]
            y = y[exist]
            u = np.diff(x)
            v = np.diff(y)
            pos_x = x[:-1] + u / 2
            pos_y = y[:-1] + v / 2
            norm = np.sqrt(u ** 2 + v ** 2)
            ax.scatter(x, y, marker='o', c=c[t//2])
            ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid", scale=10, scale_units='inches')
        # for t in range(len(data)):
        #     ax.scatter(data[t, 0::2], data[t, 1::2], marker='o', s=t*20)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(key)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        # plt.show()
        plt.savefig(Path("plots", "telaviv", key+".png"))


def plot_patches(img_arr, org_img_size, stride=None, size=None):
    """
    Plots all the patches for the first image in 'img_arr' trying to reconstruct the original image
    """

    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError('org_image_size must be a tuple')

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    fig, axes = plt.subplots(i_max, j_max)
    fig.subplots_adjust(hspace=0.01, wspace=0.05)
    jj = 0
    for i in range(i_max):
        for j in range(j_max):
            axes[i, j].imshow(img_arr[jj])
            axes[i, j].set_axis_off()
            jj += 1
    plt.savefig('test.png')


def reshape_arr(arr):
    """
    reshapes an array depending on its dimensions
    :param arr:
    :return:
    """
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return arr
        elif arr.shape[3] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])


def get_cmap(arr):
    """
    returns color space depending on dimensions of input
    :param arr:
    :return:
    """
    if arr.ndim == 3:
        return 'gray'
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return 'jet'
        elif arr.shape[3] == 1:
            return 'gray'


def zero_pad_mask(mask, desired_size):
    """
    returns a padded mask depending on desired_size
    :param mask:
    :param desired_size:
    :return:
    """
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask


def mask_to_red(mask):
    """
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    """
    c1 = mask
    c2 = np.zeros(mask.shape)
    c3 = np.zeros(mask.shape)
    c4 = mask.reshape(mask.shape)
    return np.stack((c1, c2, c3, c4), axis=-1)


def plot_semanticseg_results(org_imgs,
                             mask_imgs,
                             pred_imgs=None,
                             nm_img_to_plot=10,
                             figsize=4,
                             alpha=0.5
                             ):
    """
    Image plotting for semantic segmentation data.
    Last column is always an overlay of ground truth or prediction
    depending on what was provided as arguments.
    """
    if nm_img_to_plot > org_imgs.shape[0]:
        nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]

    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    if not (pred_imgs is None):
        cols = 4
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3

    fig, axes = plt.subplots(nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize))
    axes[0, 0].set_title("original", fontsize=15)
    axes[0, 1].set_title("ground truth", fontsize=15)
    if not (pred_imgs is None):
        axes[0, 2].set_title("prediction", fontsize=15)
        axes[0, 3].set_title("overlay", fontsize=15)
    else:
        axes[0, 2].set_title("overlay", fontsize=15)
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(mask_imgs[im_id], cmap=get_cmap(mask_imgs))
        axes[m, 1].set_axis_off()
        if not (pred_imgs is None):
            axes[m, 2].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
            axes[m, 2].set_axis_off()
            axes[m, 3].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 3].imshow(mask_to_red(pred_imgs[im_id]),
                              cmap=get_cmap(pred_imgs), alpha=alpha)
            axes[m, 3].set_axis_off()
        else:
            axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 2].imshow(mask_to_red(zero_pad_mask(mask_imgs[im_id], desired_size=org_imgs_size)),
                              cmap=get_cmap(mask_imgs), alpha=alpha)
            axes[m, 2].set_axis_off()
        im_id += 1
    fig.show()
    fig.savefig("results.png")


if __name__ == "__main__":

    # doSFM(Path("E:/University/masters/CapTracking/videos/3911a/GX011635.MP4"))
    # print("done")
    # model_name = 'scene3_batch16_lr1e4_supershuffle_noise6'
    # root_dir = Path("/disk1/yotam/capnet")
    #
    # visualize_network_performance(model_name, root_dir)

    # filter_files = ["GX011577.MP4", "GX011578.MP4", "GX011579.MP4", "GX011580.MP4",
    #                 "GX011581.MP4", "GX011582.MP4", "GX011572.MP4", "GX011573.MP4",
    #                 "GX011574.MP4", "GX011575.MP4", "GX011576.MP4", "GX011566.MP4",
    #                 "GX011567.MP4", "GX011568.MP4", "GX011569.MP4", "GX011570.MP4"]
    # db_path = Path("data", "full_db.pickle")

    db_path = Path("data", "telaviv_db.pickle")
    visualize_annotated_data(db_path, filter=None)