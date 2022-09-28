import numpy as np
from scipy.spatial.transform import Rotation as R
import math
from file_io import read_template_file
import logging
import MNI
import config


def get_curve_network_aabb(scale=1., centered=True):
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]).astype(np.float32)
        if centered:
            vertices -= np.array([0.5, 0.5, 0.5])
        vertices *= scale
        edges = np.array([
            [0, 1],
            [0, 2],
            [0, 3],
            [4, 1],
            [4, 2],
            [4, 7],
            [6, 1],
            [6, 7],
            [6, 3],
            [5, 2],
            [5, 7],
            [5, 3],

            ])
        colors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],

            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ])
        return vertices, edges, colors


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


def to_standard_coordinate_system(names, data):
    """
    given certain sticker names, converts the nx3 data to the standard coordinate system where:
    x is from left to right ear
    y is from back to front of head
    z is from bottom to top of head
    origin is defined by (x,y,z) = ((lefteye.x+righteye.x) / 2, cz.y, (lefteye.z+righteye.z) / 2)
    scale is cm. if cz is too close to origin in terms of cm, this function scales it to cm (assuming it is inch)
    note: only performs swaps, reflections, translation and possibly scale (no rotation is performed).
    :param names:
    :param data:
    :return: returns the data in the standard coordinate system
    """
    left_eye_index = names.index('lefteye')
    right_eye_index = names.index('righteye')
    cz_index = names.index('cz')
    # fpz_index = names.index('fpz')  # todo: figure out better way to know z axis. this requires user to measure fpz...
    try:
        left_triangle = names.index('left_triangle')
        right_triangle = names.index('right_triangle')
    except ValueError:
        left_triangle = names.index('fp1')
        right_triangle = names.index('fp2')
    # swap x axis with the best candidate
    x_axis = np.argmax(np.abs(data[right_eye_index] - data[left_eye_index]))
    data[:, [0, x_axis]] = data[:, [x_axis, 0]]
    # swap z axis with the best candidate (but not x)
    eyes_midpoint = ((data[left_eye_index] + data[right_eye_index]) / 2)
    fp1fp2_midpoint = ((data[left_triangle] + data[right_triangle]) / 2)
    z_axis = np.argmax(np.abs(eyes_midpoint - fp1fp2_midpoint))
    if z_axis != 0:
        data[:, [2, z_axis]] = data[:, [z_axis, 2]]

    # find reflections
    xdir = data[right_eye_index, 0] - data[left_eye_index, 0]
    ydir = data[left_eye_index, 1] - data[cz_index, 1]
    zdir = data[cz_index, 2] - data[left_eye_index, 2]
    i, j, k = (xdir > 0)*2 - 1, (ydir > 0)*2 - 1, (zdir > 0)*2 - 1
    data[:, 0] *= i
    data[:, 1] *= j
    data[:, 2] *= k

    # translate to standard origin
    eyes_midpoint = (data[right_eye_index] + data[left_eye_index]) / 2
    origin = np.array([eyes_midpoint[0], data[cz_index, 1], eyes_midpoint[2]])
    data = data - origin

    # possibly convert from inch to cm
    if data[cz_index, 2] < 7:  # distance from "middle" of brain to top is ~9-10 cm on average
        data *= 2.54
    return data


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


def affine_transform_3d_nparray(A, B):
    """
        finds best affine transformation between pc a and pc b (in terms of rmse)
        # Input: expects nx3 matrix of points
        # Returns W = the transformation to apply to A such that it matches B.
        """
    new_A = np.c_[A, np.ones(len(A))]
    new_B = np.c_[B, np.ones(len(B))]
    W = np.linalg.lstsq(new_A, new_B, rcond=None)[0]  # affine transformation matrix
    # A_transformed = np.matmul(new_A, W)
    # get_rmse(A_transformed[:, :-1], B)
    return W



def rigid_transform_3d_nparray(A, B):
    """
    finds best rigid transformation between pc a and pc b (in terms of rmse)
    # Input: expects nx3 matrix of points
    # Returns R,t = the transformation to apply to A such that it matches B.
    # R = 3x3 rotation matrix
    # t = 1x3 column vector
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A_mean = A - centroid_A
    B_mean = B - centroid_B

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
        logging.info("det(R) < R, reflection detected!, correcting for it ...\n")
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


def get_rmse(A, B):
    """
    gets rmse between 2 point clouds nx3
    :param A: nx3 point cloud
    :param B: nx3 point cloud
    :return: rmse
    """
    assert len(A.shape) == 2 and len(B.shape) == 2
    rmse = np.mean(np.linalg.norm((A - B).astype(np.float), axis=1))
    return rmse


def batch_get_rmse(A, B):
    """
    gets rmse between a point cloud nx3 to a batch of point clouds b x n x 3
    :param A:
    :param B:
    :return: b x 1 numpy array of rmse's
    """
    assert (len(A.shape) == 2 and len(B.shape) == 3) or (len(A.shape) == 3 and len(B.shape) == 2)
    A = A.astype(np.float)
    B = B.astype(np.float)
    rmse = np.mean(np.linalg.norm(A - B, axis=-1), axis=-1)
    return rmse



def fix_yaw(names, data):
    """
    given sticker names and data (nx3),
    rotates data such that x axis is along the vector going from left to right (using 6 fiducials),
    and z is pointing upwards.
    :param names:
    :param data:
    :return:
    """
    leftEye = names.index('lefteye')
    rightEye = names.index('righteye')
    leftEar = names.index('lpa')
    rightEar = names.index('rpa')
    right_triangle = names.index('right_triangle')
    left_triangle = names.index('left_triangle')
    yaw_vec_1 = (data[rightEye] - data[leftEye]) * np.array([1, 1, 0])
    yaw_vec_2 = (data[rightEar] - data[leftEar]) * np.array([1, 1, 0])
    yaw_vec_3 = (data[right_triangle] - data[left_triangle]) * np.array([1, 1, 0])
    yaw_vec_1 /= np.linalg.norm(yaw_vec_1)
    yaw_vec_2 /= np.linalg.norm(yaw_vec_2)
    yaw_vec_3 /= np.linalg.norm(yaw_vec_3)
    avg = np.mean([[yaw_vec_1], [yaw_vec_2], [yaw_vec_3]], axis=0)
    avg /= np.linalg.norm(avg)
    u = avg
    v = np.array([0, 0, 1])
    w = np.cross(v, u)
    transform = np.vstack((u, w, v))
    new_data = transform @ data.T
    return new_data.T


def from_standard_to_sim_space(names, data):
    """
    transforms data to simulation space (inverts x axis)
    :param names:
    :param data:
    :return:
    """
    data[:, 0] *= -1
    return data


def from_sim_to_standard_space(names, data):
    """
    transforms data to standard space (inverts x axis)
    :param names:
    :param data:
    :return:
    """
    return from_standard_to_sim_space(names, data)


def apply_rigid_transform(r_matrix, s_matrix, template_names, template_data, video_names, args):
    vid_est = []
    if template_names:
        names, data = template_names, template_data
    else:
        names, data, format, _ = read_template_file(args.template)
        names = names[0]
        data = data[0]
    data = to_standard_coordinate_system(names, data)
    anchor_mask = np.isin(np.array(names), np.array(config.all_possible_anchor_names))
    data_origin = data[anchor_mask]
    names_origin = np.array(names)[anchor_mask]
    data_optodes = data[~anchor_mask]
    names_optodes = np.array(names)[~anchor_mask]
    names = np.concatenate((names_origin, names_optodes)).tolist()
    for rot_mat, scale_mat in zip(r_matrix, s_matrix):
        transformed_data_sim = rot_mat @ (scale_mat @ data_optodes.T)
        data_optodes = transformed_data_sim.T
        vid_est.append([names, np.vstack((data_origin, data_optodes))])
    return vid_est


def compare_data_from_files(file_path1, file_path2, use_second_sensor):
    names1, data1, format1, _ = read_template_file(file_path1)
    names2, data2, format2, _ = read_template_file(file_path2)
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
    leftEar = names.index('lpa')
    rightEar = names.index('rpa')
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
    nosebridge = names.index('nz')
    try:
        inion = names.index('iz')
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
    # xvec = get_x_vector(names, data)
    # yvec = get_y_vector(names, data)
    # zvec = np.cross(xvec, yvec)
    # transform = np.vstack((xvec, yvec, zvec))
    # new_data = transform @ data.T
    # new_data = new_data.T
    new_data = data
    nominator = (new_data - np.min(new_data[names.index(0):], axis=0))
    denominator = (np.max(new_data[names.index(0):], axis=0) - np.min(new_data[names.index(0):], axis=0))
    new_data = nominator / denominator
    return new_data
    # xscale = new_data[names.index('rightear'), 0] - new_data[names.index('leftear'), 0]
    # yscale = new_data[names.index('nosebridge'), 1] - new_data[names.index('inion'), 1]
    # zscale = new_data[names.index('cz'), 1] - np.min(data[:, 2])


def project_sensors_to_MNI(list_of_sensor_locations, origin_optodes_names=None, resource_folder="resource", transform_anchors=False):
    """
    project new sensor locations to (statistical) MNI
    :param list_of_sensor_locations: a list of lists of [names ,data (nx3)] of all sensor locations
    :param origin_optodes_names:
    :return:
    """
    projected_locations = list_of_sensor_locations.copy()
    for i, sensor_locations in enumerate(projected_locations):
        logging.info("Projecting: {} / {} point clouds to MNI".format(i+1, len(projected_locations)))
        names = sensor_locations[0]
        data = sensor_locations[1]
        if origin_optodes_names:
            origin_selector = tuple([names.index(x) for x in origin_optodes_names])
            unsorted_origin_xyz = data[origin_selector, :]  # treated as anchors for projection (they are not changed)
            unsorted_origin_names = np.array(origin_optodes_names)
            others_selector = tuple([names.index(x) for x in names if x not in origin_optodes_names])
            others_xyz = data[others_selector, :]  # will be transformed to MNI
        else:  # last fallback, transform only non-anchors using default configuration.
            anchor_mask = np.isin(np.array(names), np.array(config.all_possible_anchor_names))
            unsorted_origin_xyz = data[anchor_mask]
            unsorted_origin_names = np.array(names)[anchor_mask]
            others_xyz = data[~anchor_mask]
        origin_xyz, selected_indices = sort_anchors(unsorted_origin_names, unsorted_origin_xyz)
        otherH, otherC, otherHSD, otherCSD, transforms = MNI.project(origin_xyz, others_xyz, selected_indices, resource_folder=resource_folder)
        # todo: should we report anything but cortex locations to caller?
        if origin_optodes_names:
            sensor_locations[1][others_selector, :] = otherH
        else:
            sensor_locations[1][~anchor_mask, :] = otherH
        if transform_anchors:
            anchors = sensor_locations[1][anchor_mask, :]
            anchors_hom = np.c_[anchors, np.ones(anchors.shape[0])]
            anchors_transformed = np.matmul(anchors_hom, transforms)
            anchors_transformed = anchors_transformed.mean(axis=0)[:, :3]
            sensor_locations[1][anchor_mask, :] = anchors_transformed  # return transformed anchors aswell (do not neccesarily reside on head surface, just naively transfromed using the affine transform)
    return projected_locations


def sort_anchors(unsorted_anchors_names: np.ndarray, unsorted_anchors_xyz: np.ndarray):
    """
    sorts anchors according to expected order from MNi projection algorithm
    :param unsorted_anchors_names: names of anchors
    :param unsorted_anchors_xyz: location of anchors
    :return: the new location of anchors and the selected indices from the possible target anchors
    """
    # these names are written in an order the projection algorithm expects (and MNI template data was written in)
    target_origin_names = np.array(["nz", "iz", "rpa", "lpa",
                                    "fp1", "fp2", "fz", "f3",
                                    "f4", "f7", "f8", "cz",
                                    "c3", "c4", "t3", "t4",
                                    "pz", "p3", "p4", "t5",
                                    "t6", "o1", "o2"])

    # sort our anchors using the order above
    selected_indices, sorting_indices = np.where(target_origin_names[:, None] == unsorted_anchors_names[None, :])
    if unsorted_anchors_xyz.ndim == 2:
        origin_xyz = unsorted_anchors_xyz[sorting_indices]
    elif unsorted_anchors_xyz.ndim == 3:
        origin_xyz = unsorted_anchors_xyz[:, sorting_indices, :]
    else:
        raise NotImplementedError
    return origin_xyz, selected_indices

def clean_model(names, data, threshold=0.3):
    """
    cleans a point cloud from points too close to each other ("almost" duplicates)
    :param names: names of the points
    :param data: the points themselves nx3
    :return: the cleaned point cloud and its names
    Note: assumes almsot duplictes only exist in unnamed points (numbers and not strings in the input names)
    """
    indices = []
    anchors = data[:names.index(0)]
    unfiltered_pc = data[names.index(0):]
    for i in range(len(unfiltered_pc)):
        for j in range(i+1, len(unfiltered_pc)):
            dist = np.linalg.norm(unfiltered_pc[i] - unfiltered_pc[j])
            if dist <= threshold:
                indices.append(j)
    selected_indices = [x for x in range(len(unfiltered_pc)) if x not in indices]
    data = np.vstack((anchors, data[tuple(selected_indices), :]))
    names = names[:names.index(0)] + [x for x in range(len(selected_indices))]
    return names, data
