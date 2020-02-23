import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "5";
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import keras
import scipy.io as sio


def visualize_pc(points1, points2=None, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if points2 is not None:
        points1 = np.transpose(points1)
        points2 = np.transpose(points2)
        names = ["Cz", "T1", "T2", "T3"]
        for i in range(len(points1)):  # plot each point + it's index as text above
            ax.scatter(points1[i, 0], points1[i, 1], points1[i, 2], color='b')
            ax.scatter(points2[i, 0], points2[i, 1], points2[i, 2], color='r')
            ax.text(points1[i, 0],
                    points1[i, 1],
                    points1[i, 2],
                    '%s' % (names[i]),
                    size=20,
                    zorder=1,
                    color='k')
            ax.text(points2[i, 0],
                    points2[i, 1],
                    points2[i, 2],
                    '%s' % (names[i]),
                    size=20,
                    zorder=1,
                    color='g')
    else:
        names = ["Nz", "Cz", "AL", "AR", "T1", "T2", "T3"]
        for i in range(len(points1)):  # plot each point + it's index as text above
            ax.scatter(points1[i, 0], points1[i, 1], points1[i, 2], color='b')
            ax.text(points1[i, 0],
                    points1[i, 1],
                    points1[i, 2],
                    '%s' % (names[i]),
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


def rigid_transform_3D(A, B):
    """
    # Input: expects 3xN matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector
    """
    assert len(A) == len(B)

    num_rows, num_cols = A.shape;

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = B.shape;
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # subtract mean
    Am = A - np.tile(centroid_A, (1, num_cols))
    Bm = B - np.tile(centroid_B, (1, num_cols))

    # dot is matrix multiplication for array
    H = Am * np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A + centroid_B

    return R, t


def get_points(file_path):
    points = []
    started = False
    with open(file_path) as fp:
        for cnt, line in enumerate(fp):
            if "Nz" in line:
                started = True
            if started:
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line[2:])
                p = [float(x) for x in numbers]
                points.append(p)
    return np.array(points)


def create_random_rotation():
    # Test with random data

    # Random rotation and translation
    R = np.mat(np.random.rand(3,3))
    t = np.mat(np.random.rand(3,1))

    # make R a proper rotation matrix, force orthonormal
    U, S, Vt = np.linalg.svd(R)
    R = U*Vt

    # remove reflection
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = U*Vt

    # number of points
    n = 4

    A = np.mat(np.random.rand(3, n));
    B = R*A + np.tile(t, (1, n))
    return A, B


def predict_from_mat(model_name, root_dir, mat_path):
    model_dir = Path.joinpath(root_dir, "models")
    best_weight_location = Path.joinpath(model_dir, "{}_best_weights.h5".format(model_name))
    model = keras.models.load_model(str(best_weight_location))
    mat_contents = sio.loadmat(mat_path)
    x = np.expand_dims(mat_contents["db"][0], axis=0)
    y_predict = model.predict(x)
    # Flip axis and switch "z" and "y" (those are the mappings in the simulation)
    rot = R.from_euler('zyx', [-y_predict[0][0], -y_predict[0][2], -y_predict[0][1]], degrees=True)
    scale = np.identity(3)
    scale[0, 0] = y_predict[0][3]
    scale[1, 1] = y_predict[0][4]
    rot_matrix = rot.as_matrix()
    rot_angels = rot.as_euler('zyx', degrees=True)
    return rot_matrix, rot_angels, scale


def calc_rmse_error(A1, A2):
    # Find the root mean squared error
    err = A1 - A2
    err = np.multiply(err, err)
    err = np.sum(err)
    return math.sqrt(err/n)


def calc_translation(A, B, rot_matrix):
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
    return -rot_matrix * centroid_A + centroid_B


if __name__ == "__main__":
    input_file_orig = Path("/disk1/yotam/capnet/new_protocol_digitizer_data/openPos18.txt")
    input_file_test = Path("/disk1/yotam/capnet/new_protocol_digitizer_data/openPos20.txt")
    output_file = Path("/disk1/yotam/capnet/new_protocol_digitizer_data/test.png")
    points_orig = get_points(input_file_orig)
    points_test = get_points(input_file_test)
    visualize_pc(points_orig, None, "baseline points")
    visualize_pc(points_test, None, "test points")
    # A, B = create_random_rotation()
    n = 4
    hat_indices = np.array([1, 4, 5, 6])
    A = np.mat(np.transpose(points_orig[hat_indices, :]))
    B = np.mat(np.transpose(points_test[hat_indices, :]))
    # Recover R and t (horn's method)
    ret_R, ret_t = rigid_transform_3D(A, B)
    rot = R.from_matrix(ret_R)
    rot_angels = rot.as_euler('zyx', degrees=True)
    # Compare the recovered R and t with the original
    recovered_B = (ret_R*A) + np.tile(ret_t, (1, n))
    visualize_pc(A, B, "Cap stickers in base vs cap stickers in Horn's recovered")
    visualize_pc(B, recovered_B, "Cap stickers in test vs cap stickers in Horn's recovered")
    rmse = calc_rmse_error(recovered_B, B)

    model_name = 'scene3_batch16_lr1e4_supershuffle_2'
    root_dir = Path("/disk1/yotam/capnet")
    gt_dir = Path.joinpath(root_dir, "ground_truth")
    db_file_orig = Path.joinpath(gt_dir, "openpos18.mat")
    db_file_test = Path.joinpath(gt_dir, "openpos20.mat")
    rmat, angels, scale = predict_from_mat(model_name, root_dir, db_file_test)
    rmat1, angels1, scale1 = predict_from_mat(model_name, root_dir, db_file_orig)
    network_t = calc_translation(A, B, rmat)
    scaled_rmat = rmat@scale
    network_scale_t = calc_translation(A, B, scaled_rmat)
    network_B = (rmat*A) + np.tile(network_t, (1, n))
    network_scale_B = (scaled_rmat*A) + np.tile(network_scale_t, (1, n))
    visualize_pc(A, network_B, "Cap stickers in base vs cap stickers from network (no scale)")
    visualize_pc(A, network_scale_B, "Cap stickers in base vs cap stickers from network (with scale)")
    visualize_pc(B, network_B, "Cap stickers in test vs cap stickers from network (no scale)")
    visualize_pc(B, network_scale_B, "Cap stickers in test vs cap stickers from network (with scale)")
    network_rmse = calc_rmse_error(network_B, B)
    network_scale_rmsee = calc_rmse_error(network_scale_B, B)
    print("RMSE (horns, network_no_scale, network_with_scale):", rmse, network_rmse, network_scale_rmsee)
    print("Euler angels (horns, network):", rot_angels, angels)

