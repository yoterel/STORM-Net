import numpy as np
from scipy.spatial.transform import Rotation as R
import re
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.metrics import mean_squared_error
import visualize
import cv2


def get_euler_angles(gt_data, model_data):
    A = np.mat(np.transpose(model_data))
    B = np.mat(np.transpose(gt_data))
    ret_R, ret_t = rigid_transform_3d(A, B)
    gt_rot_m = R.from_matrix(ret_R)
    gt_rot_e = gt_rot_m.as_euler('xyz', degrees=True)
    return gt_rot_e


def rigid_transform_svd(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1 / varP * np.sum(S)  # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c * R)

    return c, R, t


def last_rt(P, Q):
    affine_matrix = cv2.estimateAffine3D(P, Q, confidence=0.99)
    SR = affine_matrix[1][:, :-1]
    T = affine_matrix[1][:, -1]
    s = np.linalg.norm(SR, axis=1)
    S = np.identity(3) * s
    S_divide = np.copy(S)
    S_divide[S_divide == 0] = 1
    R = SR / S_divide
    return T, S, R, SR


def test():
    a1 = np.array([
        [0, 0, -1],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ])

    a2 = np.array([
        [0, 0, 2],
        [0, 0, 0],
        [0, 0, -2],
        [0, 2, 0],
        [-1, 0, 0],
    ])
    #a2 *= 2  # for testing the scale calculation
    a2 += 3  # for testing the translation calculation
    t, s, r, sr = last_rt(a1, a2)
    print("R =\n", r)
    print("c =", s)
    print("t =\n", t)
    print("Check:  a1*cR + t = a2  is", np.allclose(a1.dot(s * r) + t, a2))
    err = ((a1.dot(s * r) + t - a2) ** 2).sum()
    print("Residual error", err)


def rigid_transform_3d(A, B):
    """
    # Input: expects 3xN matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector
    """
    assert len(A) == len(B)

    num_rows, num_cols = A.shape

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = B.shape
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
        print("det(R) < R, reflection detected!, correcting for it ...\n")
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A + centroid_B

    return R, t


def get_data_from_model_file(model_path):
    points = []
    names = []
    started = False
    with open(model_path) as fp:
        for cnt, line in enumerate(fp):
            if started:
                tokens = line.split()
                name = tokens[0]
                coords = [float(x) for x in tokens[1:]]
                names.append(name)
                points.append(coords)
            if "***" in line:
                started = True

    return names, np.array(points)


def calc_rmse_error(A1, A2):
    """
    Find the root mean squared error
    :param A1: a matrix of points 3xn
    :param A2: a matrix of points 3xn
    :return: Root mean squared error between A1 and A2
    """

    err = A1 - A2
    err = np.multiply(err, err)
    err = np.sum(err)
    return math.sqrt(err/max(A1.shape))


def get_sim_data():##ed=0, n_len=0, n_dep=0, new=False):
    """
    :param ed: distance between eyes in cm
    :param n_len: nose tip distance from its origin
    :param n_dep: nose tip depth compared to eyes
    :return: base-line simulation data with x axis flipped
    """
    # my_sim_data = np.array([[ed / 2, 6, 0],  # AL
    #                         [0, 7.21 + n_dep, n_len],  # NZ
    #                         [-ed / 2, 6, 0],  # AR
    #                         [3, 8, 3.13],  # FP1
    #                         [0, 8, 5],  # FPZ
    #                         [-3, 8, 3.13],  # FP2
    #                         [0, 0, 10]  # CZ
    #                         ])
    my_new_sim_data = np.array([[2.44, 5.69, 0],
                                [0.13, 7.34, -1.21],
                                [-2.2, 6.12, 0.02],
                                [3.5, 9.75, 4.66],
                                [0.36, 9.33, 6.68],
                                [-3.77, 9.74, 4.87],
                                [-0.53, 0, 10.98]])
    my_new_sim_data += [-0.16, -2.11, 0]  # lazy to subtract mask position
    my_sim_data = my_new_sim_data
    my_sim_data[:, 0] *= -1  # flip x axis, simulator uses right hand rule
    return my_sim_data


def find_best_params(data):
    """
    :param data: locations of face & cap stickers nx3
    :return: estimated rotation & translation, and rmse error of estimation between data and simulation data
    """
    short_sim_data = get_sim_data()
    data_t = np.mat(np.transpose(data[3:]))
    sim_data_t = np.mat(np.transpose(short_sim_data[3:]))
    r, t = rigid_transform_3d(data_t, sim_data_t)
    recovered_sim_data_t = (r * data_t) + np.tile(t, (1, len(short_sim_data[3:])))
    short_rmse = calc_rmse_error(recovered_sim_data_t, sim_data_t)
    return r, t, short_rmse
    # min_rmse = np.inf
    # best_R = np.zeros((3, 3))
    # best_T = np.zeros((1, 3))
    # eye_distances = np.linspace(3.8, 5.4, num=10)
    # nose_lengths = np.linspace(-1.74, -2.24, num=10)
    # nose_depths = np.linspace(-0.5, 0.5, num=10)
    # for ed in eye_distances:
    #     for nl in nose_lengths:
    #         for nd in nose_depths:
    #             sim_data = get_sim_data(ed, nl, nd)
    #             data_t = np.mat(np.transpose(data))
    #             sim_data_t = np.mat(np.transpose(sim_data))
    #             r, t = rigid_transform_3d(data_t, sim_data_t)
    #             recovered_sim_data_t = (r * data_t) + np.tile(t, (1, len(sim_data)))
    #             rmse = calc_rmse_error(recovered_sim_data_t, sim_data_t)
    #             if rmse < min_rmse:
    #                 min_rmse, best_R, best_T, best_ed, best_nl, best_nd = rmse, r, t, ed, nl, nd
    # return best_R, best_T, best_ed, best_nl, best_nd, min_rmse


def get_sticker_data(names, data):
    if "Fp1" in names:  #newest format
        indices = [names.index("Fp1"),
                   names.index("Fpz"),
                   names.index("Fp2"),
                   names.index("Cz")
                   ]
        return data[indices, :]
    if "AL" in names:  # some old format
        indices = np.array([4, 5, 6, 1])
        return data[indices, :]
    else:  # stickers are not directly measured
        cz = (data[names.index("32"), :] + data[names.index("43"), :]) / 2
        fp1 = (data[names.index("1"), :] + data[names.index("2"), :]) / 2
        fpz = (data[names.index("3"), :] + data[names.index("7"), :]) / 2
        fp2 = (data[names.index("4"), :] + data[names.index("5"), :]) / 2
        return np.vstack((fp1, fpz, fp2, cz))


def get_face_data(names, data):
    if "Nose" in names:
        face_synonyms = ["LeftEye", "Nose", "RightEye"]
    else:
        face_synonyms = ["AL", "Nz", "AR"]
    face_indices = [names.index(face_synonyms[0]), names.index(face_synonyms[1]), names.index(face_synonyms[2])]
    face_data = data[face_indices, :]
    return face_data, face_indices


def apply_rigid_transform(r_matrix, s_matrix, model_path, gt_file, plot=True, v=1):
    names, base_model_data = get_data_from_model_file(model_path)
    face_data, face_indices = get_face_data(names, base_model_data)
    sticker_data = get_sticker_data(names, base_model_data)
    # sensor_indices = [i for i in range(len(base_model_data)) if i not in face_indices]
    # sensor_data = base_model_data[sensor_indices, :]
    r_fit, t_fit, rmse = find_best_params(np.vstack((face_data, sticker_data)))
    sim_data = get_sim_data()
    # rot_m = R.from_matrix(r_fit)
    # rot_e = rot_m.as_euler('xyz', degrees=True)

    #  get base model data to simulation space
    base_model_data_in_sim_space = (r_fit * base_model_data.T) + np.tile(t_fit, (1, len(base_model_data)))
    temp_sticker_data = get_sticker_data(names, base_model_data_in_sim_space.T)
    if plot:
        visualize.visualize_pc(np.vstack((base_model_data_in_sim_space.T[face_indices, :], temp_sticker_data)),
                     ["Left_Eye", "Nose", "Right_Eye", "CZ", "FP1", "FPZ", "FP2"],
                     sim_data,
                     ["Left_Eye", "Nose", "Right_Eye", "Cz", "FP1", "FPZ", "FP2"],
                     title="Base model data vs simulation baseline data in sim-space")

    #  apply network transformation
    transformed_base_model_data = r_matrix * (s_matrix * base_model_data_in_sim_space)
    # visualize_pc(transformed_base_model_data.T[mask_indices, :],
    #              ["Cz", "FP1", "FPZ", "FP2"],
    #              sim_data[mask_indices, :],
    #              ["Cz", "FP1", "FPZ", "FP2"],
    #              title="Prediction(base model) & simulation data in sim-space")

    # get back to real space (inverse of sim transformation)
    transformed_base_model_data_in_real_space = r_fit.T * (transformed_base_model_data - np.tile(t_fit, (1, len(base_model_data))))
    if gt_file:
        gt_names, gt_data = get_data_from_model_file(gt_file)
        gt_sticker_data = get_sticker_data(gt_names, gt_data)
        transformed_sticker_data = get_sticker_data(names, transformed_base_model_data_in_real_space.T)
        # correct for translation (useful for viz & RMSE)
        transformed_sticker_data = align_centroids(from_data=transformed_sticker_data, to_data=gt_sticker_data)
        if plot:
            visualize.visualize_pc(points_blue=transformed_sticker_data,
                         names_blue=["FP1", "FPZ", "FP2", "Cz"],
                         points_red=gt_sticker_data,
                         names_red=["FP1", "FPZ", "FP2", "Cz"],
                         title="Prediction(base model) & gt in real-space - translation corrected")
        rmse_1 = calc_rmse_error(transformed_sticker_data.T, gt_sticker_data.T)
        base_data = np.mat(np.transpose(sticker_data))
        gt_data = np.mat(np.transpose(gt_sticker_data))
        ret_R, ret_t = rigid_transform_3d(base_data, gt_data)
        if v > 1:
            print("Stickers RMSE error (prediction, gt):", rmse_1)
            print("Here is the translation between GT and base model:", ret_t)
        recovered_gt = (ret_R * base_data) + np.tile(ret_t, (1, len(base_data.T)))
        rmse_2 = calc_rmse_error(recovered_gt, gt_data)
        gt_rot_m = R.from_matrix(ret_R)
        gt_rot_e = gt_rot_m.as_euler('xyz', degrees=True)
        pred_rot_m = R.from_matrix(r_matrix)
        pred_rot_e = pred_rot_m.as_euler('xyz', degrees=True)
        if v:
            print("Euler Angles RMSE (Horns, Network):", mean_squared_error(gt_rot_e, pred_rot_e, squared=False))
        if v > 1:
            print("Horns Euler angels:", gt_rot_e)
            print("Network Euler angels:", pred_rot_e)
            print("RMSE error (horn's(baseline), gt):", rmse_2)
        if plot:
            visualize.visualize_pc(recovered_gt.T,
                         ["FP1", "FPZ", "FP2", "Cz"],
                         gt_data.T,
                         ["FP1", "FPZ", "FP2", "Cz"],
                         title="Horn's(baseline) & Ground Truth data in real-space")
        # test_rotation = [10, 8.3, 3]
        # # test_scale = np.identity(3)
        # rot = R.from_euler('xyz', test_rotation, degrees=True)
        # rot_m = rot.as_matrix()
        # transformed_base_model_data = rot_m * base_model_data_in_sim_space
        # transformed_base_model_data_in_real_space = r_fit.T * (
        #             transformed_base_model_data - np.tile(t_fit, (1, len(base_model_data))))
        # transformed_sticker_data = get_sticker_data(names, transformed_base_model_data_in_real_space.T)
        # # # correct for translation (useful for viz)
        # transformed_sticker_data = align_centroids(from_data=transformed_sticker_data, to_data=gt_sticker_data)
        # if plot:
        #     visualize.visualize_pc(points_blue=transformed_sticker_data,
        #                  names_blue=["FP1", "FPZ", "FP2", "Cz"],
        #                  points_red=gt_sticker_data,
        #                  names_red=["FP1", "FPZ", "FP2", "Cz"],
        #                  title="test")
        # print("test_rmse:", calc_rmse_error(transformed_sticker_data.T, gt_sticker_data.T))
    return transformed_base_model_data_in_real_space


def project_sensors_to_MNI(sensor_locations, v):
    return sensor_locations


def align_centroids(from_data, to_data):
    from_data = np.array(from_data)
    to_data = np.array(to_data)
    centroid_A = np.mean(from_data, axis=0)
    centroid_B = np.mean(to_data, axis=0)
    # subtract mean
    diff = centroid_A - centroid_B
    retVal = from_data - diff
    return retVal