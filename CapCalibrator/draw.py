import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from pathlib import Path
import utils
import file_io


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
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def visualize_2_pc(points_blue, names_blue=None, points_red=None, names_red=None, title=""):
    """
    plots one or two 3d point clouds in blue and red in a nx3 numpy array format
    :param points_blue:
    :param names_blue:
    :param points_red:
    :param names_red:
    :param title:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if points_red is not None:
        for i in range(len(points_blue)):  # plot each point + it's index as text above
            ax.scatter(points_blue[i, 0], points_blue[i, 1], points_blue[i, 2], color='b')
            if names_blue:
                ax.text(points_blue[i, 0],
                        points_blue[i, 1],
                        points_blue[i, 2],
                        '%s' % (names_blue[i]),
                        size=20,
                        zorder=1,
                        color='k')
        for i in range(len(points_red)):
            ax.scatter(points_red[i, 0], points_red[i, 1], points_red[i, 2], color='r')
            if names_red:
                ax.text(points_red[i, 0],
                        points_red[i, 1],
                        points_red[i, 2],
                        '%s' % (names_red[i]),
                        size=20,
                        zorder=1,
                        color='g')
    else:
        for i in range(len(points_blue)):  # plot each point + it's index as text above
            ax.scatter(points_blue[i, 0], points_blue[i, 1], points_blue[i, 2], color='b')
            if names_blue:
                ax.text(points_blue[i, 0],
                        points_blue[i, 1],
                        points_blue[i, 2],
                        '%s' % (names_blue[i]),
                        size=20,
                        zorder=1,
                        color='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(30, 30)
    ax.set_title(title)
    plt.show()
    # plt.savefig(output_file)


def plot_3d_pc(ax, data, selected, names=None):
    """
    plots a 3d point cloud representation of data
    used in experiment viewer
    :param ax: the axis to plot into
    :param data: the data (nx3 numpy array)
    :param selected: an int representing the currently selected data point - will be painted red
    :param names: the names of the data points
    :return:
    """
    if not names:
        names = [str(i) for i in range(len(data))]
    colors = ['b'] * len(data)
    # colors[selected] = 'r'
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


def visualize_annotated_data(db_path, synth_data_path, filter=None):
    """
    plots sticker locations over 10 frames from a real database and closest output from a synthetic database folder for comparison.
    the locations are centered and plotted in normalized screen space
    :param synth_data_path:
    :param db_path:
    :param filter:
    :return:
    """
    db = file_io.load_db(db_path, "pickle", filter)
    synth_db = file_io.load_raw_json_db(synth_data_path)
    synth_db = synth_db[0]  # selects data, discards label
    utils.center_data(synth_db)
    shift = 0
    for key in db.keys():
        data = db[key][shift]["data"][0]
        if len(db[key][shift]["data"][0].shape) < 3:
            data = np.expand_dims(data, axis=0)
        data[:, :, 0::2] /= 960
        data[:, :, 1::2] /= 540
        utils.center_data(data)
        data = np.squeeze(data)
        closest_synth_image = np.argmin(np.sum((synth_db - data)**2, axis=(1, 2)))
        selected_synth_data = synth_db[closest_synth_image]
        # s_linear = [n for n in range(len(data))]
        gen_and_save_quiver_plot(Path("plots", "telaviv", key+".png"), data)
        gen_and_save_quiver_plot(Path("plots", "telaviv", key+"_synth.png"), selected_synth_data)


def gen_and_save_quiver_plot(my_path, data, title=None):
    fig, ax = plt.subplots()
    c = ['b', 'b', 'b', 'r', 'r', 'r', 'g']
    for t in range(0, data.shape[1], 2):
        x = data[:, t]
        y = data[:, t + 1]
        exist = np.nonzero(x)
        x = x[exist]
        y = y[exist]
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u / 2
        pos_y = y[:-1] + v / 2
        norm = np.sqrt(u ** 2 + v ** 2)
        ax.scatter(x, y, marker='o', c=c[t // 2])
        ax.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid", scale=10, scale_units='inches')
    # for t in range(len(data)):
    #     ax.scatter(data[t, 0::2], data[t, 1::2], marker='o', s=t*20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if title:
        ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # plt.show()
    plt.savefig(my_path)
    plt.close(fig)


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


def visualize_network_performance(model_name, root_dir):
    """
    This function was used to asses performance of STORM-net during research, can be removed in future
    :param model_name:
    :param root_dir:
    :return:
    """
    #############################################################
    db_path = Path("data", "full_db.pickle")
    gt = file_io.load_db(db_path)
    vid_name = "GX011578.MP4"
    synth_data_dir = Path.joinpath(root_dir, "captures_special")
    pickle_file_path = Path.joinpath(synth_data_dir, "data.pickle")
    if not pickle_file_path.is_file():
        X, Y = file_io.load_db(synth_data_dir)
        x_train, x_val, y_train, y_val, x_test, y_test = utils.split_data(X, Y)
        file_io.serialize_data(pickle_file_path, x_train, x_val, y_train, y_val, x_test, y_test)
    else:
        x_train, x_val, y_train, y_val, x_test, y_test = file_io.deserialize_data(pickle_file_path)
    A = gt[vid_name]["data"][0]
    A[:, 0::2] /= 960
    A[:, 1::2] /= 540
    temp = np.copy(A[:, 12:14])
    A[:, 12:14] = A[:, 6:8]
    A[:, 6:8] = temp
    min_index = 0
    min_dist = np.inf
    for i in range(len(x_train)):
        B = x_train[i]
        dist = np.linalg.norm(A - B)
        if dist < min_dist:
            min_dist = dist
            min_index = i
    data_dir = Path.joinpath(root_dir, "scene3_100k")
    model_dir = Path.joinpath(root_dir, "models")
    best_weight_location = Path.joinpath(model_dir, model_name)
    model, _ = file_io.load_clean_keras_model(best_weight_location)
    gt_pred = model.predict(np.expand_dims(A, axis=0))
    y_predict_special = model.predict(np.expand_dims(x_train[min_index], axis=0))
    print("gt pred: ", gt_pred, gt[vid_name]["label"])
    print("synth pred: ", y_predict_special, y_train[min_index])
    filter = [vid_name]
    visualize_annotated_data(db_path, filter)
    filter = ["image_{:05d}.json".format(min_index)]
    visualize_annotated_data(synth_data_dir, filter)
#################################################################
    pickle_file_path = Path.joinpath(data_dir, "data.pickle")
    x_train, x_val, y_train, y_val, x_test, y_test = file_io.deserialize_data(pickle_file_path)
    fig = plt.figure()
    ax = plt.axes()
    n, bins, patches = ax.hist(y_train[:, 0], 50, density=True, facecolor='r', alpha=0.75)
    n, bins, patches = ax.hist(y_train[:, 1], 50, density=True, facecolor='b', alpha=0.75)
    n, bins, patches = ax.hist(y_train[:, 2], 50, density=True, facecolor='g', alpha=0.75)
    ax.set_xlabel('Angle')
    ax.set_ylabel('# of instances')
    ax.set_title("Histogram of angle distribution in training set")
    plt.savefig('plots/angle_dist.png')
    y_predict = model.predict(x_test)
    mean_results = np.mean(abs(y_predict - y_test), 0)
    print("err_x:", mean_results[0])
    print("err_y:", mean_results[1])
    print("err_z:", mean_results[2])
    ############################################################################
    # index = np.argmin(np.mean(abs(y_predict-y_test), axis=1))
    # baseline_data = x_test[index, :, :]
    # gt = y_test[index, :]
    preds = []
    for i in range(x_test.shape[1]):
        baseline = np.copy(x_test)
        baseline[:, i, :] = np.zeros((x_test.shape[0], x_test.shape[2]))
        pred = model.predict(baseline)
        result = np.mean(np.mean(abs(pred - y_test), axis=1))
        preds.append(result)
    fig = plt.figure()
    ax = plt.axes()
    x = list(range(10))
    ax.plot(x, preds)
    ax.set_xlabel('Index of missing frame')
    ax.set_ylabel('Error')
    ax.set_title("Error as a function of index of missing frame")
    # plt.show()
    plt.savefig('plots/frames_err.png')
    ############################################################################
    preds = []
    gt = y_test
    for i in range(0, x_test.shape[2], 2):
        baseline = np.copy(x_test)
        baseline[:, :, i:i+2] = np.zeros((baseline.shape[0], baseline.shape[1], 2))
        pred = model.predict(baseline)
        result = np.mean(np.mean(abs(pred - gt), axis=1))
        preds.append(result)
    fig = plt.figure()
    ax = plt.axes()
    x = list(range(x_test.shape[2] // 2))
    ax.plot(x, preds)
    ax.set_xlabel('Index of missing sticker')
    ax.set_ylabel('Error')
    ax.set_title("Error as a function of index of missing sticker")
    # plt.show()
    plt.savefig('plots/sticker_err.png')
    ############################################################################
    sticker_occurrences = []
    for i in range(0, x_test.shape[2], 2):
        occurrence = np.count_nonzero(x_test[:, :, i:i+2]) // 2
        sticker_occurrences.append(occurrence / (x_test.shape[0]*x_test.shape[1]))
    fig = plt.figure()
    ax = plt.axes()
    x = list(range(x_test.shape[2] // 2))
    ax.plot(x, sticker_occurrences)
    ax.set_xlabel('Index of sticker')
    ax.set_ylabel('Occurrences percentage')
    ax.set_title("Occurrences percentage of every sticker")
    # plt.show()
    plt.savefig('plots/sticker_percent.png')
    print("done")


def plot_skull_vs_error(skull, intra_digi, intra_vid, inter):
    plt.scatter(skull, intra_digi, s=100)
    plt.scatter(skull, intra_vid, s=100)
    plt.scatter(skull, inter, s=100)
    plt.legend(['Intra-Method Error Digiziter', 'Intra-Method Error Ours', 'Inter-Method Error'])
    plt.ylabel('RMSE Error [cm]')
    plt.xlabel('Skull Size [cm]')
    plt.show()
