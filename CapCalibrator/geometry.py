import numpy as np
from scipy.spatial.transform import Rotation as R
import re
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.metrics import mean_squared_error
import visualize
import cv2
from file_io import read_template_file
from pathlib import Path


def align_centroids(a, b):
    """
    aligns a point cloud to another such that the first one's centroid is moved to the second one's centroid
    :param a: the first pc (nx3)
    :param b: the second pc (nx3)
    :return: point cloud a moved to b's centroid
    """
    a = np.array(a)
    b = np.array(b)
    centroid_a = np.mean(a, axis=0)
    centroid_b = np.mean(b, axis=0)
    # subtract mean
    diff = centroid_a - centroid_b
    return a - diff


def check_right_handed_system(names, data):
    """
    given certain sticker names, checks weather the nx3 data is represented in a right handed coordinate system
    :param names:
    :param data:
    :return: returns which axis should be flipped to turn into a right handed coordinate system
    """
    leftEye = names.index('lefteye')
    rightEye = names.index('righteye')
    CZ = names.index('cz')
    xdir = data[rightEye, 0] - data[leftEye, 0]
    ydir = data[leftEye, 1] - data[CZ, 1]
    zdir = data[CZ, 2] - data[leftEye, 2]
    if zdir > 0:
        if (xdir > 0 and ydir > 0) or (xdir < 0 and ydir < 0):
            return [1, 1, 1]
        else:
            return [-1, 1, 1]
    else:
        # print("Warning ! Z axis is pointing downwards, fixing.")
        if (xdir > 0 and ydir > 0) or (xdir < 0 and ydir < 0):
            return [1, 1, -1]
        else:
            if xdir > 0:
                return [1, -1, -1]
            else:
                return [-1, 1, -1]


def get_euler_angles(gt_data, model_data):
    """
    given two point clouds, returns the euler angles required to rotate point cloud a to point cloud b
    note: returned rotation transformation is best in terms of least squares error.
    :param gt_data:
    :param model_data:
    :return:
    """
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


def rigid_transform_3d_nparray(A, B):
    """
    finds best (in terms of rmse) rigid transformation between pc a and pc b
    # Input: expects 3XN matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 1x3 column vector
    """
    A_transpose = A.T
    B_transpose = B.T
    centroid_A = np.mean(A_transpose, axis=0)
    centroid_B = np.mean(B_transpose, axis=0)

    A_mean = A_transpose - centroid_A
    B_mean = B_transpose - centroid_B

    H = A_mean.T @ B_mean
    U, S, Vt = np.linalg.svd(H)

    flip = np.linalg.det(Vt.T @ U.T)
    ones = np.identity(len(Vt))
    ones[-1, -1] = flip
    R = Vt.T @ ones @ U.T
    t = centroid_B - R @ centroid_A
    return R, t


def rigid_transform_3d(A, B):
    """
    finds best (in terms of rmse) rigid transformation between pc a and pc b
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


def calc_rmse_error(A1, A2):
    """
    Finds the root mean squared error between two point clouds
    :param A1: a matrix of points 3xn
    :param A2: a matrix of points 3xn
    :return: Root mean squared error between A1 and A2
    """

    err = A1 - A2
    err = np.multiply(err, err)
    err = np.sum(err)
    return math.sqrt(err/max(A1.shape))


def get_sim_data(model_data):##ed=0, n_len=0, n_dep=0, new=False):
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
    tx = (model_data[0, 0] + model_data[2, 0]) / 2
    ty = model_data[6, 1]
    tz = (model_data[0, 2] + model_data[2, 2]) / 2
    sim_data = model_data - np.array([tx, ty, tz])
    sim_data[:, 0] *= -1
    return sim_data

    # my_new_sim_data = np.array([[2.44, 5.69, 0],
    #                             [0.13, 7.34, -1.21],
    #                             [-2.2, 6.12, 0.02],
    #                             [3.5, 9.75, 4.66],
    #                             [0.36, 9.33, 6.68],
    #                             [-3.77, 9.74, 4.87],
    #                             [-0.53, 0, 10.98]])
    # my_new_sim_data[3:] += [-0.16, -2.11, 0]  # lazy to subtract mask position
    # my_sim_data = my_new_sim_data
    # my_sim_data[:, 0] *= -1  # flip x axis, simulator uses right hand rule
    # return my_sim_data


def find_best_params(data):
    """
    estimates rotation & translation, and rmse error of estimation between data and simulation data
    :param data: locations of face & cap stickers nx3
    :return:
    """
    short_sim_data = get_sim_data(data)
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
    names, base_model_data = read_template_file(model_path)
    face_data, face_indices = get_face_data(names, base_model_data)
    sticker_data = get_sticker_data(names, base_model_data)
    # sensor_indices = [i for i in range(len(base_model_data)) if i not in face_indices]
    # sensor_data = base_model_data[sensor_indices, :]
    fiducials_data = np.vstack((face_data, sticker_data))
    r_fit, t_fit, rmse = find_best_params(fiducials_data)
    # rot_m = R.from_matrix(r_fit)
    # rot_e = rot_m.as_euler('xyz', degrees=True)

    #  get base model data to simulation space
    base_model_data_in_sim_space = (r_fit * base_model_data.T) + np.tile(t_fit, (1, len(base_model_data)))
    temp_sticker_data = get_sticker_data(names, base_model_data_in_sim_space.T)
    if plot:
        sim_data = get_sim_data(fiducials_data)
        visualize.visualize_pc(np.vstack((base_model_data_in_sim_space.T[face_indices, :], temp_sticker_data)),
                               ["Left_Eye", "Nose", "Right_Eye", "FP1", "FPZ", "FP2", "CZ"],
                               sim_data,
                               ["Left_Eye", "Nose", "Right_Eye", "FP1", "FPZ", "FP2", "Cz"],
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
        gt_names, gt_data = read_template_file(gt_file)
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


def get_rmse(A, B):
    """
    gets rmse between 2 point clouds nx3
    :param A: nx3 point cloud
    :param B: nx3 point cloud
    :return: rmse
    """
    diff = np.abs(A - B)
    mse = np.mean(diff*diff)
    rmse = np.sqrt(mse)
    return rmse

# find template-digitizer error by rigidly transforming template (using fiducials) x2
# find template-digitizer error by rigidly transforming template (using all stickers) x2
# compare inter-session digitizer (using given origin)
# compare outer-session digitizer (x2)
# compare inter-session digitizer (using 2nd sensor as origin)
# compare outer-session digitizer (x2, using 2nd sensor as origin)
#


def compare_data_from_files(file_path1, file_path2, use_second_sensor):
    names1, data1, format1 = read_template_file(file_path1)
    names2, data2, format2 = read_template_file(file_path2)
    assert(names1 == names2)
    assert(len(data1) == len(data2))
    index_of_spiral1 = names1.index(0)
    index_of_spiral2 = names2.index(0)
    if use_second_sensor:
        spiral_data1 = data1[index_of_spiral1:, 0, :] - data1[index_of_spiral1:, 1, :]
        spiral_data2 = data2[index_of_spiral2:, 0, :] - data2[index_of_spiral2:, 1, :]
    else:
        spiral_data1 = data1[index_of_spiral1:, 0, :]
        spiral_data2 = data2[index_of_spiral2:, 0, :]
    return get_rmse(spiral_data1, spiral_data2)


def get_x_vector(names, data):
    leftEye = names.index('lefteye')
    rightEye = names.index('righteye')
    leftEar = names.index('leftear')
    rightEar = names.index('rightear')
    Fp2 = names.index('fp2')
    Fp1 = names.index('fp1')
    yaw_vec_1 = (data[rightEye] - data[leftEye]) * np.array([1, 1, 0])
    yaw_vec_2 = (data[rightEar] - data[leftEar]) * np.array([1, 1, 0])
    yaw_vec_3 = (data[Fp1] - data[Fp2]) * np.array([1, 1, 0])
    yaw_vec_1 /= np.linalg.norm(yaw_vec_1)
    yaw_vec_2 /= np.linalg.norm(yaw_vec_2)
    yaw_vec_3 /= np.linalg.norm(yaw_vec_3)
    avg = np.mean([[yaw_vec_1], [yaw_vec_2], [yaw_vec_3]], axis=0)
    avg /= np.linalg.norm(avg)
    return avg


def get_y_vector(names, data):
    nosebridge = names.index('nosebridge')
    try:
        inion = names.index('inion')
        yvec = data[nosebridge] - data[inion]
    except ValueError:
        spiral = data[names.index(0):, :]
        yvec = data[nosebridge] - ((spiral[84] + spiral[83]) / 2)
    yvec /= np.linalg.norm(yvec)
    return yvec


def normalize_coordinates(names, data):
    """
    normalizes data according to the following method:
    right handed coordinate system, scaled from 0 to 1 in all axis (excluding face fiducials)
    note: xyz are chosen such that "front" is a vector from inion to nasion, "right" is from left ear to right ear,
    and "up" is the cross between them (in a good brain this points upwards towards cz from center of brain).
    :param names:
    :param data:
    :return:
    """
    xvec = get_x_vector(names, data)
    yvec = get_y_vector(names, data)
    zvec = np.cross(xvec, yvec)
    transform = np.vstack((xvec, yvec, zvec))
    new_data = transform @ data.T
    new_data = new_data.T
    nominator = (new_data - np.min(new_data[names.index(0):], axis=0))
    denominator = (np.max(new_data[names.index(0):], axis=0) - np.min(new_data[names.index(0):], axis=0))
    new_data = nominator / denominator
    return new_data
    # xscale = new_data[names.index('rightear'), 0] - new_data[names.index('leftear'), 0]
    # yscale = new_data[names.index('nosebridge'), 1] - new_data[names.index('inion'), 1]
    # zscale = new_data[names.index('cz'), 1] - np.min(data[:, 2])


def print_calibration_results(path_to_template, experiment_folder_path):
    template_names, template_data, template_format = read_template_file(path_to_template)
    flip_axis = check_right_handed_system(template_names, template_data)
    template_data[:, 0] *= flip_axis[0]
    template_data[:, 1] *= flip_axis[1]
    template_data[:, 2] *= flip_axis[2]
    template_data = normalize_coordinates(template_names, template_data)  # normalize data
    template_face_data = template_data[(template_names.index('lefteye'),
                                       template_names.index('righteye'),
                                       template_names.index('nosetip'),
                                       template_names.index('fp1'),
                                       template_names.index('fp2'),
                                       template_names.index('fpz'),
                                       template_names.index('cz')), :]
    template_spiral_data = template_data[template_names.index(0):, :]  # select spiral
    for file in experiment_folder_path.glob('*'):
        if "raw" not in file.name:
            file_names, file_data, file_format = read_template_file(file)
            file_data = file_data[:, 0, :] - file_data[:, 1, :]
            flip_axis = check_right_handed_system(file_names, file_data)
            file_data[:, 0] *= flip_axis[0]
            file_data[:, 1] *= flip_axis[1]
            file_data[:, 2] *= flip_axis[2]
            file_data = normalize_coordinates(file_names, file_data)  # normalize data
            file_face_data = file_data[(file_names.index('lefteye'),
                                       file_names.index('righteye'),
                                       file_names.index('nosetip'),
                                       file_names.index('fp1'),
                                       file_names.index('fp2'),
                                       file_names.index('fpz'),
                                       file_names.index('cz')), :]
            file_spiral_data = file_data[file_names.index(0):, :]  # select first sensor and begin from spiral
            ret_R, ret_t = rigid_transform_3d_nparray(template_face_data.T, file_face_data.T)
            # ret_R1, ret_t1 = rigid_transform_3d(np.asmatrix(template_face_data.T), np.asmatrix(file_face_data.T))
            estimation = (ret_R @ template_spiral_data.T).T + ret_t
            rmse = get_rmse(estimation, file_spiral_data)
            print("template-digitizer error (transform using fiducials):", rmse)
            # vis_estimation = (ret_R @ template_face_data.T).T + ret_t
            # visualize.visualize_pc(points_blue=vis_estimation,
            #                        points_red=file_face_data,
            #                        title="test")
            ret_R, ret_t = rigid_transform_3d_nparray(template_spiral_data.T, file_spiral_data.T)
            estimation = (ret_R @ template_spiral_data.T).T + ret_t
            rmse = get_rmse(estimation, file_spiral_data)
            print("template-digitizer (transform using all-sensors):", rmse)
    path1 = Path.joinpath(experiment_folder_path, "yaara1.txt")
    path2 = Path.joinpath(experiment_folder_path, "yaara2.txt")
    rmse = compare_data_from_files(path1, path2, use_second_sensor=False)
    print("digitizer-digitzier (not using 2nd sensor) rmse:", rmse)
    rmse = compare_data_from_files(path1, path2, use_second_sensor=True)
    print("digitizer-digitzier (using 2nd sensor) rmse:", rmse)

