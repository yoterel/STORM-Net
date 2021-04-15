import numpy as np
import geometry
from file_io import read_template_file, read_digitizer_multi_noptodes_experiment_file
import re
import logging
from pathlib import Path
from scipy.stats import pearsonr, ttest_rel
from scipy.spatial.transform import Rotation as R


def reproduce_experiments(r_matrix, s_matrix, video_names, args):
    """
    reproduces original experiments reported in manuscript, results are printed to log or plotted where applicable
    :param r_matrix: see caller
    :param s_matrix: see caller
    :param video_names: see caller
    :param args: see caller
    :return: -
    """
    do_digi_error_experiment()

    dig_ses1, dig_ses2, sessions = do_dig2dig_experiment(args.template, args.ground_truth)

    vid_ses1, vid_ses2 = do_vid2vid_project_beforeMNI_experiment(args.template, video_names, r_matrix, s_matrix)
    # vid_ses1, vid_ses2 = do_vid2vid_project_afterMNI_experiment(args.template, video_names, r_matrix, s_matrix)

    do_vid2dig_experiment(dig_ses1, dig_ses2, vid_ses1, vid_ses2)

    do_histogram_experiment(dig_ses1, dig_ses2, vid_ses1, vid_ses2)

    do_old_experiment(r_matrix, s_matrix, video_names, args)


def do_digi_error_experiment():
    """
    reproduces experiment where various number of optodes were measured with digitizer
    tries to find relationship between number of optodes and error received
    :return:
    """
    data = read_digitizer_multi_noptodes_experiment_file("resource/digi_error_exp.txt")
    errors = {}
    for datum in data:
        number_of_optodes = len(datum) // 10
        error = []
        for exp in range(5):
            real_exp = exp * number_of_optodes * 2
            begin = datum[real_exp:real_exp + number_of_optodes]
            end = datum[real_exp + number_of_optodes:real_exp + 2*number_of_optodes]
            error.append(geometry.get_rmse(begin, end))
        error = np.array(error)
        error = np.mean(error)
        errors.setdefault(number_of_optodes, []).append(error)
    for key, value in errors.items():
        logging.info("{}->{:.3f}".format(key, value[0]))


def do_dig2dig_experiment(template_path, experiment_folder):
    """
    tests how accurate is the digitizer between 2 sessions after MNI projection
    :param experiment_folder:
    :return:
    """
    template_names, template_data, _, _ = read_template_file(template_path)
    template_names = template_names[0]
    template_data = template_data[0]
    template_data = geometry.to_standard_coordinate_system(template_names, template_data)

    experiment_file_list = []
    sessions = [[], []]

    origin = ["leftear", "rightear", "nosebridge", "cz"]
    others = ["fp1", "fp2", "fpz", "o1", "o2", "oz", "f7", "f8"]
    others_template = template_names[template_names.index(0):]
    digi_names = origin + others
    combined_others = others + others_template
    full_names = origin + combined_others
    if experiment_folder.is_dir():
        for exp_file in sorted(experiment_folder.glob("*.txt")):
            experiment_file_list.append(exp_file)
    else:
        experiment_file_list.append(experiment_folder)
    for exp_file in experiment_file_list:
        logging.info(exp_file.name)
        file_names, file_data, file_format, skull = read_template_file(exp_file)
        for i, session in enumerate(zip(file_names[:2], file_data[:2])):
            names = session[0]
            data = session[1][:, 0, :] - session[1][:, 1, :]  # subtracts second sensor
            data = geometry.to_standard_coordinate_system(names, data)
            origin_selector = tuple([names.index(x) for x in origin])
            subject_specific_origin = data[origin_selector, :]
            others_selector = tuple([names.index(x) for x in others])
            subject_specific_others = data[others_selector, :]
            # others_selector_template = tuple([template_names.index(x) for x in others])
            # W = geometry.affine_transform_3d_nparray(template_data[others_selector_template, :],
            #                                                                  subject_secific_others)
            # others_selector_template = tuple([template_names.index(x) for x in others_template])
            # others_data = template_data[others_selector_template, :]
            # new_others_data = np.c_[others_data, np.ones(len(others_data))]
            # new_others_data_transformed = np.matmul(new_others_data, W)[:,:-1]
            # subject_secific_others = np.vstack((subject_secific_others, new_others_data_transformed))
            sessions[i].append([digi_names, np.vstack((subject_specific_origin, subject_specific_others))])
    cached_result_ses1 = "cache/session1_digi_MNI"
    cached_result_ses2 = "cache/session2_digi_MNI"
    if not Path(cached_result_ses1 + ".npy").is_file() or not Path(cached_result_ses2 + ".npy").is_file():
        digi_projected_ses1 = geometry.project_sensors_to_MNI(sessions[0], origin)
        digi_ss_data_ses1 = np.array([x[1] for x in digi_projected_ses1], dtype=np.object)
        np.save(cached_result_ses1, digi_ss_data_ses1)
        digi_projected_ses2 = geometry.project_sensors_to_MNI(sessions[1], origin)
        digi_ss_data_ses2 = np.array([x[1] for x in digi_projected_ses2], dtype=np.object)
        np.save(cached_result_ses2, digi_ss_data_ses2)
    else:
        digi_ss_data_ses1 = np.load(cached_result_ses1 + ".npy", allow_pickle=True)
        digi_ss_data_ses2 = np.load(cached_result_ses2 + ".npy", allow_pickle=True)
    digi_projected_ses1_others = digi_ss_data_ses1[:, digi_names.index(others[0]):, :]
    digi_projected_ses2_others = digi_ss_data_ses2[:, digi_names.index(others[0]):, :]

    # selector_template = tuple([template_names.index(x) for x in others])
    # to_fill = np.empty((digi_projected_ses1_others.shape[0],
    #                     digi_projected_ses1_others.shape[1] + len(others_template),
    #                     digi_projected_ses1_others.shape[2]))
    # for i in range(to_fill.shape[0]):
    #     W1 = geometry.affine_transform_3d_nparray(template_data[selector_template, :],
    #                                               digi_projected_ses1_others[i].astype(np.float))
    #     others_selector_template = tuple([template_names.index(x) for x in others_template])
    #     others_data = template_data[others_selector_template, :]
    #     new_others_data = np.c_[others_data, np.ones(len(others_data))]
    #     new_others_data_transformed = np.matmul(new_others_data, W1)[:, :-1]
    #     to_fill[i] = np.vstack((digi_projected_ses1_others[i], new_others_data_transformed))
    # # digi_projected_ses1_others = to_fill.copy()
    # for i in range(to_fill.shape[0]):
    #     W2 = geometry.affine_transform_3d_nparray(template_data[selector_template, :],
    #                                               digi_projected_ses2_others[i].astype(np.float))
    #     others_selector_template = tuple([template_names.index(x) for x in others_template])
    #     others_data = template_data[others_selector_template, :]
    #     new_others_data = np.c_[others_data, np.ones(len(others_data))]
    #     new_others_data_transformed = np.matmul(new_others_data, W2)[:, :-1]
    #     to_fill[i] = np.vstack((digi_projected_ses2_others[i], new_others_data_transformed))
    # # digi_projected_ses2_others = to_fill.copy()

    errors = []
    for i in range(len(sessions[0])):
        rmse_error = geometry.get_rmse(digi_projected_ses1_others[i], digi_projected_ses2_others[i])
        errors.append(rmse_error)
    rmse_error_mean = np.mean(np.array(errors))
    rmse_error_std = np.std(np.array(errors))
    logging.info("dig2dig mean, std rmse (after MNI projection): {:.3f}, {:.3f}".format(rmse_error_mean,
                                                                                        rmse_error_std))

    # from draw import visualize_2_pc
    # visualize_2_pc(points_blue=digi_projected_ses1_others[0], points_red=digi_projected_ses2_others[0])
    return [others, digi_projected_ses1_others],\
           [others, digi_projected_ses2_others],\
            sessions


def do_vid2vid_project_afterMNI_experiment(template_path, video_names, r_matrices, s_matrices):
    names, data, format, _ = read_template_file(template_path)
    names = names[0]
    data = data[0]
    data = geometry.to_standard_coordinate_system(names, data)
    output_others_names = ["fp1", "fp2", "fpz", "o1", "o2", "oz", "f7", "f8"]
    others_names = output_others_names + names[names.index(0):]
    origin_names = ["leftear", "rightear", "nosebridge", "cz"]
    list_extension = [x for x in others_names]
    new_names = origin_names + list_extension
    sessions = [[], []]
    rots = [[], []]
    for i, (rot_mat, scale_mat, vid) in enumerate(zip(r_matrices, s_matrices, video_names)):
        subject_name, session_name = vid.split("_")
        session_number = int(re.findall(r'\d+', session_name)[0]) - 1
        if session_number == 2:
            continue
        origin_selector = tuple([names.index(x) for x in origin_names])
        origin = data[origin_selector, :]
        others_selector = tuple([names.index(x) for x in others_names])
        others = data[others_selector, :]
        sessions[session_number].append([new_names, np.vstack((origin, others))])
        rots[session_number].append(rot_mat)
    cached_result_ses1 = "cache/session1_vid_MNI"
    cached_result_ses2 = "cache/session2_vid_MNI"
    if not Path(cached_result_ses1 + ".npy").is_file() or not Path(cached_result_ses2 + ".npy").is_file():
        vid_projected_ses1 = geometry.project_sensors_to_MNI(sessions[0], origin_names)
        vid_ss_data_ses1 = np.array([x[1] for x in vid_projected_ses1], dtype=np.object)
        np.save(cached_result_ses1, vid_ss_data_ses1)
        vid_projected_ses2 = geometry.project_sensors_to_MNI(sessions[1], origin_names)
        vid_ss_data_ses2 = np.array([x[1] for x in vid_projected_ses2], dtype=np.object)
        np.save(cached_result_ses2, vid_ss_data_ses2)
    else:
        vid_ss_data_ses1 = np.load(cached_result_ses1 + ".npy", allow_pickle=True)
        vid_ss_data_ses2 = np.load(cached_result_ses2 + ".npy", allow_pickle=True)
    vid_projected_ses1_others = vid_ss_data_ses1[:, new_names.index(output_others_names[0]):, :]
    vid_projected_ses2_others = vid_ss_data_ses2[:, new_names.index(output_others_names[0]):, :]
    vid_ses1_transformed = vid_projected_ses1_others.copy()
    vid_ses2_transformed = vid_projected_ses2_others.copy()
    for i, rot in enumerate(zip(rots[0], rots[1])):
        vid_ses1_transformed[i, :, :] = (rot[0] @ (vid_projected_ses1_others[i, :, :].T)).T
        vid_ses2_transformed[i, :, :] = (rot[1] @ (vid_projected_ses2_others[i, :, :].T)).T
    errors = []
    for i in range(len(sessions[0])):
        rmse_error = geometry.get_rmse(vid_ses1_transformed[i], vid_ses2_transformed[i])
        errors.append(rmse_error)
    rmse_error_mean = np.mean(np.array(errors))
    rmse_error_std = np.std(np.array(errors))
    logging.info("vid2vid mean, std rmse (after MNI projection): {:.3f}, {:.3f}".format(rmse_error_mean,
                                                                                        rmse_error_std))
    output_selector = tuple([others_names.index(x) for x in output_others_names])
    return [output_others_names, vid_ses1_transformed[:, output_selector, :]],\
           [output_others_names, vid_ses2_transformed[:, output_selector, :]]


def do_vid2vid_project_beforeMNI_experiment(template_path, video_names, r_matrices, s_matrices, digi_sessions=None):
    """
    applies transform from network before MNI projection
    :param template_path:
    :param video_names:
    :param r_matrices:
    :param s_matrices:
    :param digi_sessions: pass this to use anchors from digitizer
    :return:
    """
    names, data, format, _ = read_template_file(template_path)
    names = names[0]
    data = data[0]
    data = geometry.to_standard_coordinate_system(names, data)
    output_others_names = ["fp1", "fp2", "fpz", "o1", "o2", "oz", "f7", "f8"]
    others_names = output_others_names + names[names.index(0):]
    origin_names = ["leftear", "rightear", "nosebridge", "cz"]
    new_names = origin_names + others_names
    sessions = [[], []]
    for i, (rot_mat, scale_mat, vid) in enumerate(zip(r_matrices, s_matrices, video_names)):
        subject_name, session_name = vid.split("_")
        session_number = int(re.findall(r'\d+', session_name)[0]) - 1
        if session_number == 2:
            continue
        origin_selector = tuple([names.index(x) for x in origin_names])
        origin = data[origin_selector, :]
        others_selector = tuple([names.index(x) for x in others_names])
        others = data[others_selector, :]
        if digi_sessions:
            digi_origin_selector = tuple([digi_sessions[0][0][0].index(x) for x in origin_names])
            digi_origin = digi_sessions[session_number][i//3][1][digi_origin_selector, :]
            W = geometry.affine_transform_3d_nparray(origin, digi_origin)
            others = np.matmul(np.c_[others, np.ones(len(others))], W)[:,:-1]
            # r, t = geometry.rigid_transform_3d_nparray(digi_origin, origin)
            # digi_origin = (r @ (digi_origin.T)).T + t
            # others = (r @ others.T).T + t
            origin = digi_origin
        others = (rot_mat @ others.T).T  # apply transform before MNI projection
        sessions[session_number].append([new_names, np.vstack((origin, others))])
    cached_result_ses1 = "cache/session1_vid_MNI_transb"
    cached_result_ses2 = "cache/session2_vid_MNI_transb"
    if not Path(cached_result_ses1 + ".npy").is_file() or not Path(cached_result_ses2 + ".npy").is_file():
        vid_projected_ses1 = geometry.project_sensors_to_MNI(sessions[0], origin_names)
        vid_ss_data_ses1 = np.array([x[1] for x in vid_projected_ses1], dtype=np.object)
        np.save(cached_result_ses1, vid_ss_data_ses1)
        vid_projected_ses2 = geometry.project_sensors_to_MNI(sessions[1], origin_names)
        vid_ss_data_ses2 = np.array([x[1] for x in vid_projected_ses2], dtype=np.object)
        np.save(cached_result_ses2, vid_ss_data_ses2)
    else:
        vid_ss_data_ses1 = np.load(cached_result_ses1 + ".npy", allow_pickle=True)
        vid_ss_data_ses2 = np.load(cached_result_ses2 + ".npy", allow_pickle=True)
    vid_projected_ses1_others = vid_ss_data_ses1[:, new_names.index(output_others_names[0]):, :]
    vid_projected_ses2_others = vid_ss_data_ses2[:, new_names.index(output_others_names[0]):, :]
    errors = []
    for i in range(len(sessions[0])):
        rmse_error = geometry.get_rmse(vid_projected_ses1_others[i], vid_projected_ses2_others[i])
        errors.append(rmse_error)
    rmse_error_mean = np.mean(np.array(errors))
    rmse_error_std = np.std(np.array(errors))
    logging.info("vid2vid mean, std rmse (after MNI projection): {:.3f}, {:.3f}".format(rmse_error_mean,
                                                                                        rmse_error_std))
    output_selector = tuple([others_names.index(x) for x in output_others_names])
    return [output_others_names, vid_projected_ses1_others[:, output_selector, :]],\
           [output_others_names, vid_projected_ses2_others[:, output_selector, :]]


def do_vid2dig_experiment(digi_ses1, digi_ses2, vid_ses1, vid_ses2):
    inter_method_rmse1 = [geometry.get_rmse(x, y) for x, y in zip(digi_ses1[1], vid_ses1[1])]
    inter_method_rmse2 = [geometry.get_rmse(x, y) for x, y in zip(digi_ses1[1], vid_ses2[1])]
    inter_method_rmse3 = [geometry.get_rmse(x, y) for x, y in zip(digi_ses2[1], vid_ses1[1])]
    inter_method_rmse4 = [geometry.get_rmse(x, y) for x, y in zip(digi_ses2[1], vid_ses2[1])]
    inter_method_rmse_avg = np.mean([inter_method_rmse1, inter_method_rmse2, inter_method_rmse3, inter_method_rmse4])
    inter_method_rmse_std = np.std([inter_method_rmse1, inter_method_rmse2, inter_method_rmse3, inter_method_rmse4])
    logging.info("dig2vid mean, std rmse (after MNI projection): {:.3f}, {:.3f}".format(inter_method_rmse_avg,
                                                                                        inter_method_rmse_std))


def do_histogram_experiment(dig_ses1, dig_ses2, vid_ses1, vid_ses2):
    """
    loads and displays histogram figure for manuscript
    :return:
    """
    from draw import plot_histogram
    # try:
    #     import scipy.io as sio
    #     dig2dig_after_MNI = sio.loadmat("resource/dig2dig.mat")["dig2digDist"]
    #     dig2vid_after_MNI = sio.loadmat("resource/dig2vid.mat")["dig2vidDist"]
    #     vid2vid_after_MNI = sio.loadmat("resource/vid2vid.mat")["vid2vidDist"]
    #     plot_histogram(dig2dig_after_MNI, dig2vid_after_MNI, vid2vid_after_MNI)
    # except Exception:
    # logging.info("tried to reproduce figure but pkl data is missing")
    dig2dig_after_MNI = np.linalg.norm(dig_ses1[1].astype(np.float) - dig_ses2[1].astype(np.float), axis=2).T
    dig2vid_after_MNI0 = np.linalg.norm(vid_ses1[1].astype(np.float) - dig_ses1[1].astype(np.float), axis=2).T
    dig2vid_after_MNI1 = np.linalg.norm(vid_ses1[1].astype(np.float) - dig_ses2[1].astype(np.float), axis=2).T
    dig2vid_after_MNI2 = np.linalg.norm(vid_ses2[1].astype(np.float) - dig_ses1[1].astype(np.float), axis=2).T
    dig2vid_after_MNI3 = np.linalg.norm(vid_ses2[1].astype(np.float) - dig_ses2[1].astype(np.float), axis=2).T
    dig2vid_after_MNI = (dig2vid_after_MNI0 + dig2vid_after_MNI1 + dig2vid_after_MNI2 + dig2vid_after_MNI3) / 4
    vid2vid_after_MNI = np.linalg.norm(vid_ses1[1].astype(np.float) - vid_ses2[1].astype(np.float), axis=2).T
    plot_histogram(dig2dig_after_MNI, dig2vid_after_MNI, vid2vid_after_MNI)


def get_digi2digi_results(path_to_template, experiment_folder_path, rot_as_matrix=False, spiral_output_type="orig"):
    """
    returns digitizer results from experiements
    :param path_to_template:
    :param experiment_folder_path:
    :param rot_as_matrix:
    :param spiral_output_type:
    :return:
    """
    template_names, template_data, template_format, _ = read_template_file(path_to_template)
    template_data = template_data[0]
    template_names = template_names[0]
    template_data = geometry.to_standard_coordinate_system(template_names, template_data)
    # template_data = normalize_coordinates(template_names, template_data)  # normalize data
    template_mask_data = template_data[(template_names.index('fp1'),
                                        template_names.index('fp2'),
                                        template_names.index('fpz'),
                                        template_names.index('cz'),
                                        template_names.index('o1'),
                                        template_names.index('o2'),
                                        template_names.index('oz'),
                                        template_names.index('f7'),
                                        template_names.index('f8')), :]
    template_face_data = template_data[(template_names.index('leftear'),
                                        template_names.index('rightear'),
                                        template_names.index('lefteye'),
                                        template_names.index('righteye'),
                                        template_names.index('nosebridge'),
                                        template_names.index('nosetip')), :]
    template_spiral_data = template_data[template_names.index(0):, :]  # select spiral

    experiment_file_list = []
    skulls = []
    spiral_output = []
    optode_estimations = {}
    rot_estimations = {}
    if experiment_folder_path.is_dir():
        for exp_file in sorted(experiment_folder_path.glob("*.txt")):
            experiment_file_list.append(exp_file)
    else:
        experiment_file_list.append(experiment_folder_path)

    for exp_file in experiment_file_list:
        logging.info(exp_file.name)
        file_names, file_data, file_format, skull = read_template_file(exp_file)
        skull_sizes = []
        for i, session in enumerate(zip(file_names, file_data)):
            if not session[0]:
                optode_estimations.setdefault(exp_file.stem, []).append(None)
                rot_estimations.setdefault(exp_file.stem, []).append(None)
                continue
            names = session[0]
            data = session[1][:, 0, :] #- session[1][:, 1, :]  # subtract second sensor
            data = geometry.to_standard_coordinate_system(names, data)
            # file_data = normalize_coordinates(file_names, file_data)  # normalize data
            file_mask_data = data[(names.index('fp1'),
                                   names.index('fp2'),
                                   names.index('fpz'),
                                   names.index('cz'),
                                   names.index('o1'),
                                   names.index('o2'),
                                   names.index('oz'),
                                   names.index('f7'),
                                   names.index('f8')), :]
            file_face_data = data[(names.index('leftear'),
                                   names.index('rightear'),
                                   names.index('lefteye'),
                                   names.index('righteye'),
                                   names.index('nosebridge'),
                                   names.index('nosetip')), :]
            # align faces as intermediate step
            ret_R, ret_t = geometry.rigid_transform_3d_nparray(template_face_data, file_face_data)
            aligned_template_mask = (ret_R @ template_mask_data.T).T + ret_t

            if spiral_output_type == "orig":
                subject_secific_origin = data[(names.index('leftear'),
                                               names.index('rightear'),
                                               names.index('nosebridge'),
                                               names.index('cz')), :]
                subject_secific_spiral = file_mask_data
                spiral_output.append([subject_secific_origin, subject_secific_spiral])
            elif spiral_output_type == "spiral_transformed":
                if i == 0:
                    subject_secific_origin = data[(names.index('leftear'),
                                                   names.index('rightear'),
                                                   names.index('nosebridge'),
                                                   names.index('cz')), :]
                    subject_secific_spiral = (ret_R @ template_spiral_data.T).T + ret_t
                    spiral_output.append([subject_secific_origin, subject_secific_spiral])

            # find mask transform
            ret_R, ret_t = geometry.rigid_transform_3d_nparray(aligned_template_mask, file_mask_data)
            # vis_estimation = (ret_R @ template_mask_data.T).T + ret_t
            # draw.visualize_2_pc(points_blue=vis_estimation,
            #                     points_red=file_mask_data,
            #                     title="test")
            gt_rot_m = R.from_matrix(ret_R)
            gt_rot_e = gt_rot_m.as_euler('xyz', degrees=True)
            logging.info("Digitizer Euler angels: " + str(gt_rot_e))
            # note: we apply only the mask transformation and report that to downstream.
            # facial alignment was an intermediate result
            estimation = (ret_R @ template_spiral_data.T).T
            optode_estimations.setdefault(exp_file.stem, []).append(estimation)
            if rot_as_matrix:
                rot_estimations.setdefault(exp_file.stem, []).append(ret_R)
            else:
                rot_estimations.setdefault(exp_file.stem, []).append(gt_rot_e)
            skull_sizes.append([np.linalg.norm(data[(names.index('leftear'))] - data[(names.index('rightear'))]),
                                np.linalg.norm(data[(names.index('fpz'))] - data[(names.index('o1'))])])
        skulls.append(np.mean(np.array(skull_sizes), axis=0) / 2)

        # rmse = geometry.get_rmse(estimation, file_spiral_data)
        # print("template-digitizer error (transform using fiducials):", rmse)
        # vis_estimation = (ret_R @ template_face_data.T).T + ret_t
        # visualize.visualize_pc(points_blue=vis_estimation,
        #                        points_red=file_face_data,
        #                        title="test")
        # ret_R, ret_t = rigid_transform_3d_nparray(template_spiral_data.T, file_spiral_data.T)
        # estimation = (ret_R @ template_spiral_data.T).T + ret_t
        # rmse = geometry.get_rmse(estimation, file_spiral_data)
        # print("template-digitizer (transform using all-sensors):", rmse)
        # rmse = geometry.get_rmse(estimations[0], estimations[1])
        # print("digi2digi rmse between session 1 and session 2:", rmse)
        # rmse = geometry.get_rmse(estimations[0], estimations[2])
        # print("digi2digi rmse between session 1 and session 3:", rmse)
        # rmse = geometry.get_rmse(estimations[1], estimations[2])
        # print("digi2digi rmse between session 2 and session 3:", rmse)
    return optode_estimations, rot_estimations, skulls, spiral_output


def do_old_experiment(r_matrix, s_matrix, video_names, args):
    digi2digi_est, digi2digi_rot, skull_radii, ss_data = get_digi2digi_results(args.template,
                                                                               args.ground_truth,
                                                                               True)
    vid2vid_est = []
    names, data, format, _ = read_template_file(args.template)
    names = names[0]
    data = data[0]
    data = geometry.to_standard_coordinate_system(names, data)
    data_spiral = data[names.index(0):, :]  # select spiral

    digi_intra_method_sessions1 = []
    digi_intra_method_sessions2 = []
    digi_rot_sessions1 = []
    digi_rot_sessions2 = []
    digi_rot_sessions3 = []

    vid_rot_sessions1 = []
    vid_rot_sessions2 = []
    vid_rot_sessions3 = []

    digi_r_matrix = []

    # calculate all errors for manuscript
    for i, (rot_mat, scale_mat, vid) in enumerate(zip(r_matrix, s_matrix, video_names)):
        subject_name, session_name = vid.split("_")
        session_number = int(re.findall(r'\d+', session_name)[0]) - 1

        transformed_data_sim = rot_mat @ (scale_mat @ data_spiral.T)
        vid2vid_est.append(transformed_data_sim.T)
        rot = R.from_matrix(rot_mat)
        try:
            digi_rot = R.from_matrix(digi2digi_rot[subject_name][session_number])
            digi_rot_euler = digi_rot.as_euler('xyz', degrees=True)
        except Exception:
            digi_rot_euler = None

        if session_number == 0:
            digi_intra_method_sessions1.append(digi2digi_est[subject_name][session_number])
            digi_rot_sessions1.append(digi_rot_euler)
            vid_rot_sessions1.append(rot.as_euler('xyz', degrees=True))
            digi_r_matrix.append(digi2digi_rot[subject_name][session_number])
        if session_number == 1:
            digi_intra_method_sessions2.append(digi2digi_est[subject_name][session_number])
            digi_rot_sessions2.append(digi_rot_euler)
            vid_rot_sessions2.append(rot.as_euler('xyz', degrees=True))
            digi_r_matrix.append(digi2digi_rot[subject_name][session_number])
        if session_number == 2:
            digi_rot_sessions3.append(digi_rot_euler)
            vid_rot_sessions3.append(rot.as_euler('xyz', degrees=True))

    digi_intra_method_rmse = [geometry.get_rmse(x, y) for x, y in
                              zip(digi_intra_method_sessions1, digi_intra_method_sessions2)]
    digi_intra_method_rmse_avg = np.mean(digi_intra_method_rmse)
    digi_intra_method_rmse_std = np.std(digi_intra_method_rmse)
    logging.info(
        "digi2digi rmse avg, std: {:.3f}, {:.3f}".format(digi_intra_method_rmse_avg, digi_intra_method_rmse_std))

    vid_intra_method_sessions1 = vid2vid_est[::3]
    vid_intra_method_sessions2 = vid2vid_est[1::3]
    vid_intra_method_rmse = [geometry.get_rmse(x, y) for x, y in
                             zip(vid_intra_method_sessions1, vid_intra_method_sessions2)]
    vid_intra_method_rmse_avg = np.mean(vid_intra_method_rmse)
    vid_intra_method_rmse_std = np.std(vid_intra_method_rmse)
    logging.info("vid2vid rmse avg, std: {:.3f}, {:.3f}".format(vid_intra_method_rmse_avg, vid_intra_method_rmse_std))

    inter_method_rmse1 = [geometry.get_rmse(x, y) for x, y in
                          zip(digi_intra_method_sessions1, vid_intra_method_sessions1)]
    inter_method_rmse2 = [geometry.get_rmse(x, y) for x, y in
                          zip(digi_intra_method_sessions1, vid_intra_method_sessions2)]
    inter_method_rmse3 = [geometry.get_rmse(x, y) for x, y in
                          zip(digi_intra_method_sessions2, vid_intra_method_sessions1)]
    inter_method_rmse4 = [geometry.get_rmse(x, y) for x, y in
                          zip(digi_intra_method_sessions2, vid_intra_method_sessions2)]
    inter_method_rmse_avg = np.mean([inter_method_rmse1, inter_method_rmse2, inter_method_rmse3, inter_method_rmse4])
    inter_method_rmse_std = np.std([inter_method_rmse1, inter_method_rmse2, inter_method_rmse3, inter_method_rmse4])
    logging.info("digi2vid rmse avg, std: {:.3f}, {:.3f}".format(inter_method_rmse_avg, inter_method_rmse_std))

    # if we are missing a session, remove it from other experiments too
    pop_indices = [i for i, x in enumerate(digi_rot_sessions3) if x is None]
    for index in pop_indices:
        digi_rot_sessions1.pop(index)
        digi_rot_sessions2.pop(index)
        digi_rot_sessions3.pop(index)
        vid_rot_sessions1.pop(index)
        vid_rot_sessions2.pop(index)
        vid_rot_sessions3.pop(index)

    # calc t-test statistics
    a_vid = np.array(vid_intra_method_rmse)
    a_dig = np.array(digi_intra_method_rmse)
    a_inter = np.mean(np.array([inter_method_rmse1, inter_method_rmse2, inter_method_rmse3, inter_method_rmse4]),
                      axis=0)

    # do skulls plot
    import draw
    skull_radii = np.array(skull_radii)
    circumferences = 2 * np.pi * np.sqrt((skull_radii[:, 0] ** 2 + skull_radii[:, 1] ** 2) / 2)
    draw.plot_skull_vs_error(circumferences, digi_intra_method_rmse, vid_intra_method_rmse, a_inter)
    logging.info("skull circumferences mean: {:.3f}".format(np.mean(circumferences)))
    logging.info("skull circumferences std: {:.3f}".format(np.std(circumferences)))
    t1, p1 = ttest_rel(a_vid, a_dig)
    logging.info("t-test results intra (t, p): {:.3f}, {:.3f}".format(t1, p1))
    t2, p2 = ttest_rel(a_dig, a_inter)
    logging.info("t-test results inter (t, p): {:.3f}, {:.3f}".format(t2, p2))

    # calc shifts in every direction for each method
    digi2digi_shift = np.array(
        [z - (x + y) / 2 for x, y, z in zip(digi_rot_sessions1, digi_rot_sessions2, digi_rot_sessions3)])
    vid2vid_shift = np.array(
        [z - (x + y) / 2 for x, y, z in zip(vid_rot_sessions1, vid_rot_sessions2, vid_rot_sessions3)])

    # correlate the shifts
    x_corr_new, px = pearsonr(digi2digi_shift[:, 0], vid2vid_shift[:, 0])
    y_corr_new, py = pearsonr(digi2digi_shift[:, 1], vid2vid_shift[:, 1])
    z_corr_new, pz = pearsonr(digi2digi_shift[:, 2], vid2vid_shift[:, 2])

    logging.info(
        "Shift correlation (x, px, y, py, z, pz):"
        "{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(x_corr_new, px, y_corr_new, py, z_corr_new, pz))

    # do MNI plots
    dig_est_ses1 = []
    dig_est_ses2 = []
    data_others = [x[1] for x in ss_data]
    data_origin = [x[0] for x in ss_data]
    list_extension = [x for x in range(len(data_others))]
    digi_names = ["leftear", "rightear", "nosebridge", "cz"] + list_extension
    if not Path("cache/session1_digi_MNI_ss.npy").is_file() or not Path("cache/session2_digi_MNI_ss.npy").is_file():
        # digi2digi session1: calc error using subject-specific MNI anchors
        for i in range(0, len(data_origin), 3):
            dig_est_ses1.append([digi_names, np.vstack((data_origin[i], data_others[i]))])
            dig_est_ses2.append([digi_names, np.vstack((data_origin[i + 1], data_others[i + 1]))])

        digi_projected_data_ses1 = geometry.project_sensors_to_MNI(dig_est_ses1)
        digi = np.array([x[1] for x in digi_projected_data_ses1], dtype=np.object)
        np.save("cache/session1_digi_MNI_ss", digi)
        digi_projected_data_ses2 = geometry.project_sensors_to_MNI(dig_est_ses2)
        digi = np.array([x[1] for x in digi_projected_data_ses2], dtype=np.object)
        np.save("cache/session2_digi_MNI_ss", digi)

    do_MNI = False
    if do_MNI:
        # everything else (digi2digi, vid2vid, digi2vid) all sessions: calc error using template MNI anchors
        vid_est = []
        dig_est = []
        names, data, format, _ = read_template_file(args.template)
        names = names[0]
        data = data[0]
        data = geometry.to_standard_coordinate_system(names, data)
        if 0 in names:
            data_origin = data[:names.index(0), :]  # non numbered optodes are not calibrated
            data_optodes = data[names.index(0):, :]  # selects optodes for applying calibration
        else:
            data_origin = data
            data_optodes = np.zeros(3)
        for rot_mat, scale_mat in zip(r_matrix, s_matrix):
            transformed_data_sim = rot_mat @ (scale_mat @ data_optodes.T)
            vid_est.append([names, np.vstack((data_origin, transformed_data_sim.T))])
        for rot_mat, scale_mat in zip(digi_r_matrix, s_matrix):
            transformed_data_sim = rot_mat @ (scale_mat @ data_optodes.T)
            dig_est.append([names, np.vstack((data_origin, transformed_data_sim.T))])
        from scipy import io

        # io.savemat('names.mat',
        #            {'names': np.array([vid_est[0][0]], dtype=np.object)})

        # session1_vid = np.array([x[1] for x in vid_est[0::3]], dtype=np.object)
        # io.savemat('session1_vid_no_MNI.mat',
        #            {'session1_vid_no_MNI': session1_vid})
        # session2_vid = np.array([x[1] for x in vid_est[1::3]], dtype=np.object)
        # io.savemat('session2_vid_no_MNI.mat',
        #            {'session2_vid_no_MNI': session2_vid})
        # session1_dig = np.array([x[1] for x in dig_est[0::2]], dtype=np.object)
        # io.savemat('session1_digi_no_MNI.mat',
        #            {'session1_digi_no_MNI': session1_dig})
        # session2_dig = np.array([x[1] for x in dig_est[1::2]], dtype=np.object)
        # io.savemat('session2_digi_no_MNI.mat',
        #            {'session2_digi_no_MNI': session2_dig})

        vid_projected_data_session1 = geometry.project_sensors_to_MNI(vid_est[0::3])
        session1_vid = np.array([x[1] for x in vid_projected_data_session1], dtype=np.object)
        np.save("cache/session1_vid_MNI", session1_vid)
        # io.savemat('session1_vid.mat',
        #            {'session1_vid': session1_vid})
        vid_projected_data_session2 = geometry.project_sensors_to_MNI(vid_est[1::3])
        session2_vid = np.array([x[1] for x in vid_projected_data_session2], dtype=np.object)
        np.save("cache/session2_vid_MNI", session2_vid)
        # io.savemat('session2_vid.mat',
        #            {'session2_vid': session2_vid})
        digi_projected_data_session1 = geometry.project_sensors_to_MNI(dig_est[0::2])
        session1_dig = np.array([x[1] for x in digi_projected_data_session1], dtype=np.object)
        np.save("cache/session1_digi_MNI", session1_dig)
        # io.savemat('session1_digi.mat',
        #            {'session1_digi': session1_dig})
        digi_projected_data_session2 = geometry.project_sensors_to_MNI(dig_est[1::2])
        session2_dig = np.array([x[1] for x in digi_projected_data_session2], dtype=np.object)
        np.save("cache/session2_digi_MNI", session2_dig)
        # io.savemat('session2_digi.mat',
        #            {'session2_digi': session2_dig})

    digi_ss = np.load("cache/session1_digi_MNI_ss.npy", allow_pickle=True)
    digi_template = np.load("cache/session2_digi_MNI_ss.npy", allow_pickle=True)
    digi_ss_spiral = digi_ss[:, digi_names.index(0):, :]
    digi_template_spiral = digi_template[:, digi_names.index(0):, :]
    errors = []
    for i in range(len(digi_ss_spiral)):
        rmse_error = geometry.get_rmse(digi_ss_spiral[i], digi_template_spiral[i])
        errors.append(rmse_error)
    rmse_error_f = np.mean(np.array(errors))
    logging.info("with ss mni vs without: {:.3f}".format(rmse_error_f))
    draw.visualize_2_pc(points_blue=digi_ss_spiral[0], points_red=digi_ss_spiral[1])
    print("done!")