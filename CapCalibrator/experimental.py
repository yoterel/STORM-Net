import numpy as np
import geometry
from file_io import read_template_file, read_digitizer_multi_noptodes_experiment_file
import re
import logging
from pathlib import Path
from scipy.stats import pearsonr, ttest_rel
from scipy.spatial.transform import Rotation as R
import predict
import torch_src.MNI_torch as MNI_torch
import torch
import draw


def do_network_robustness_test(sticker_locations, args):
    """
    reports results of network robustness to various noises introduced to annotation
    :param sticker_locations: the original locations as marked by the user / AI in the GUI
    :param args: args
    :return:
    """
    robustness_file = Path("cache/robustness.npy")
    noise_mag_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if robustness_file.is_file():
        t_r_rmse = np.load(robustness_file)
    else:
        orig_r_matrix, orig_s_matrix = predict.predict_rigid_transform(np.copy(sticker_locations), None, args)
        orig_euler = [R.from_matrix(x).as_euler('xyz', degrees=True) for x in orig_r_matrix]
        t_r_rmse = []
        t_s_rmse = []
        b_shape = sticker_locations.shape[0]
        t_shape = sticker_locations.shape[1]
        for j in range(len(noise_mag_list)):
            r_rmse = []
            s_rmse = []
            for i in range(10):
                sticker_locations_copy = np.copy(sticker_locations)
                another_view = np.reshape(sticker_locations_copy, (b_shape, t_shape, sticker_locations_copy.shape[-1] // 2, 2))
                zero_locs = np.where(another_view == np.array([0, 0]))
                noise_mag = noise_mag_list[j]
                noise_shift = noise_mag / 2
                noise = (np.random.random_sample(sticker_locations_copy.shape) * noise_mag) - noise_shift
                sticker_locations_copy += noise
                another_view[zero_locs] = 0
                r_matrix, s_matrix = predict.predict_rigid_transform(sticker_locations_copy, None, args)
                euler = [R.from_matrix(x).as_euler('xyz', degrees=True) for x in r_matrix]
                r_rmse.append(np.linalg.norm(np.array(orig_euler) - np.array(euler), axis=0))
                s_rmse.append(np.linalg.norm(np.array(orig_s_matrix) - np.array(s_matrix)))
            t_r_rmse.append(np.mean(r_rmse, axis=0))
            t_s_rmse.append(np.mean(s_rmse))
        t_r_rmse = np.array(t_r_rmse)
        np.save(robustness_file, t_r_rmse)
    draw.plot_robustness(np.array(noise_mag_list), t_r_rmse)


def do_opt2dig_experiment(digi_ses1, digi_ses2, grid_search, rots, scales, video_names):
    """
    reports results given grid search on parameter space and digitizer sessions
    (and rots / scales usedto achieve optimum)
    :param digi_ses1: digitizer session1
    :param digi_ses2: digitizer session2
    :param grid_search: template model with many experiments where optodes were rotated and scaled
    :param rots: the rots used for grid_search
    :param scales: the scales used grid_search
    :param video_names: names of videos for printing
    :return:
    """
    subject_names = [x.split("_")[0] for x in video_names[::3]]
    # net_rots_ses1 = [x for x in net_rots[::3]]
    # net_rots_ses2 = [x for x in net_rots[1::3]]
    errors_ses1 = []
    errors_ses2 = []
    real_errors_ses1 = []
    real_errors_ses2 = []
    cross_error_12 = []
    cross_error_21 = []
    # ses1 = opt_ses1.copy()
    # ses2 = opt_ses1.copy()
    # output_scales = []
    for i in range(len(digi_ses1[1])):
        inter_method_rmse1 = geometry.batch_get_rmse(digi_ses1[1][i], grid_search)
        inter_method_rmse2 = geometry.batch_get_rmse(digi_ses2[1][i], grid_search)
        # inter_method_rmse1 = [geometry.get_rmse(x, digi_ses1[1][i]) for x in opt_ses1[1]]
        # inter_method_rmse2 = [geometry.get_rmse(x, digi_ses2[1][i]) for x in opt_ses2[1]]
        errors_ses1.append(inter_method_rmse1)
        errors_ses2.append(inter_method_rmse2)
    for i, (error1, error2) in enumerate(zip(errors_ses1, errors_ses2)):
        min_error_index1 = np.argmin(error1).astype(np.int)
        min_error1 = error1[min_error_index1]
        logging.info("session: {}, subject: {}, min error: {}, rot: {}, scale: {}".format(1,
                                                                                          subject_names[i],
                                                                                          min_error1,
                                                                                          rots[min_error_index1],
                                                                                          scales[min_error_index1]
                                                                                          ))
        # logging.info("network rot: {}".format(R.from_matrix(net_rots_ses1[i]).as_euler('xyz', degrees=True)))
        real_errors_ses1.append([min_error_index1, min_error1])
        # ses1[1][i] = opt_ses1[1][min_error_index]
        # output_scales.append(scales[min_error_index])
        min_error_index2 = np.argmin(error2).astype(np.int)
        min_error2 = error2[min_error_index2]
        logging.info("session: {}, subject: {}, min error: {}, rot: {}, scale: {}".format(2,
                                                                                          subject_names[i],
                                                                                          min_error2,
                                                                                          rots[min_error_index2],
                                                                                          scales[min_error_index2]
                                                                                          ))
        # logging.info("network rot: {}".format(R.from_matrix(net_rots_ses2[i]).as_euler('xyz', degrees=True)))
        real_errors_ses2.append([min_error_index2, min_error2])

        rmse12 = geometry.get_rmse(grid_search[min_error_index1], digi_ses2[1][i])
        logging.info("session: {}, subject: {}, min error: {}, rot: {}, scale: {}".format("1->2",
                                                                                          subject_names[i],
                                                                                          rmse12,
                                                                                          rots[min_error_index1],
                                                                                          scales[min_error_index1]
                                                                                          ))
        cross_error_12.append(rmse12)
        rmse21 = geometry.get_rmse(grid_search[min_error_index2], digi_ses1[1][i])
        logging.info("session: {}, subject: {}, min error: {}, rot: {}, scale: {}".format("2->1",
                                                                                          subject_names[i],
                                                                                          rmse21,
                                                                                          rots[min_error_index2],
                                                                                          scales[min_error_index2]
                                                                                          ))
        cross_error_21.append(rmse21)
        # ses2[1][i] = opt_ses2[1][min_error_index]
        # output_scales.append(scales[min_error_index])
        # output_scales.append(np.ones(3))
    # ses1[1] = ses1[1][:10]
    # ses2[1] = ses2[1][:10]
    logging.info("ses1 optimal inter-validation error: {}".format(np.mean([x[1] for x in real_errors_ses1])))
    logging.info("ses2 optimal inter-validation error: {}".format(np.mean([x[1] for x in real_errors_ses2])))
    logging.info("ses2 optimal error with ses1 parameters: {}".format(np.mean(cross_error_12)))
    logging.info("ses1 optimal error with ses2 parameters: {}".format(np.mean(cross_error_21)))
    # return sessions incase caller wants to plot / visualize
    # output_scales = [np.diag(x) for x in output_scales]
    # return ses1, ses2, output_scales


def do_parameter_grid_search_experiment(opt):
    """
    experiement to use an exhaustive search of parameter space to find optimal rotation and scale of template model
    applies transform from before MNI projection.
    :param opt command line options
    :return:
    """
    random_grid_space_size = 500000
    splits = 5
    names, data, format, _ = read_template_file(opt.template)
    names = names[0]
    data = data[0]
    data = geometry.to_standard_coordinate_system(names, data)
    output_others_names = ["fp1", "fp2", "fpz", "o1", "o2", "oz", "f7", "f8"]
    others_names = output_others_names #+ names[names.index(0):]
    origin_names = ["lpa", "rpa", "nz", "cz"]
    full_names = origin_names + others_names
    cached_result_ses1 = "cache/session1_opt_MNI"
    recreate_files = False
    for i in range(splits):
        name1 = cached_result_ses1 + str(i)
        if not Path(name1 + ".npy").is_file():
            recreate_files = True
    ses1_list = []
    if recreate_files:
        # perform random grid search
        sessions = []
        np.random.seed(42)
        if Path("cache/rots").is_file() and Path("cache/scales").is_file():
            rots = np.load("cache/rots.npy", allow_pickle=True)
            scales = np.load("cache/scales.npy", allow_pickle=True)
        else:
            rx = np.random.rand(random_grid_space_size) * 14 - 7
            ry = np.random.rand(random_grid_space_size) * 14 - 7
            rz = np.random.rand(random_grid_space_size) * 14 - 7
            sx = np.random.rand(random_grid_space_size) * 0.6 + 0.7
            sy = np.random.rand(random_grid_space_size) * 0.6 + 0.7
            sz = np.random.rand(random_grid_space_size) * 0.6 + 0.7
            scales = np.array([sx, sy, sz]).T
            rots = np.array([rx, ry, rz]).T
            # save grid search parameters
            np.save("cache/rots", rots)
            np.save("cache/scales", scales)
        from scipy.spatial.transform import Rotation as R
        rot_grid_search = [R.from_euler('xyz', rots[i], degrees=True).as_matrix() for i in range(len(rots))]
        scale_grid_search = [np.diag(scales[i]) for i in range(len(scales))]
        origin_selector = tuple([names.index(x) for x in origin_names])
        others_selector = tuple([names.index(x) for x in others_names])
        origin = data[origin_selector, :]
        others = data[others_selector, :]
        # apply rotation and scale (batched version)
        transformed_others = (np.array(rot_grid_search) @ (np.array(scale_grid_search) @ others.T)).transpose(0, 2, 1)
        # for rot, scale in zip(rot_grid_search, scale_grid_search):
        #
        #     others = (rot @ (scale @ others.T)).T  # apply transform before MNI projection
        #     sessions.append([full_names, np.vstack((origin, others))])
        for split in range(splits):
            min_range = (random_grid_space_size // splits)*(split)
            max_range = (random_grid_space_size // splits)*(split+1)
            # project to MNI every split
            # all_sensors = np.array([sessions[i][1] for i in range(min_range, max_range)])
            anchors = origin  # same anchors for all experiments...
            sensors = transformed_others[min_range:max_range, :, :]
            sorted_anchors, indices = geometry.sort_anchors(np.array(origin_names), anchors)
            device = opt.device
            origin_xyz_torch = torch.from_numpy(sorted_anchors).float().to(device)
            sensors_xyz_torch = torch.from_numpy(sensors).float().to(device)
            selected_indices_torch = torch.from_numpy(indices).to(device)
            torch_mni, _, _ = MNI_torch.torch_project_non_differentiable(origin_xyz_torch,
                                                                         sensors_xyz_torch,
                                                                         selected_indices_torch,
                                                                         output_errors=False)

            # vid_projected_ses1 = geometry.project_sensors_to_MNI(sessions[0][min_range:max_range], origin_names)
            # vid_ss_data_ses1 = np.array([x[1] for x in vid_projected_ses1], dtype=np.object)
            np_mni = torch_mni.cpu().numpy()
            np.save(cached_result_ses1+str(split), np_mni)
            ses1_list.append(np_mni)
    else:
        rots = np.load("cache/rots.npy", allow_pickle=True)
        scales = np.load("cache/scales.npy", allow_pickle=True)
        for i in range(splits):
            name1 = cached_result_ses1 + str(i)
            ses1_list.append(np.load(name1 + ".npy", allow_pickle=True))
    vid_ss_data_ses1 = np.vstack(ses1_list).astype(np.float)
    # output_selector = tuple([full_names.index(x) for x in output_others_names])
    return output_others_names, vid_ss_data_ses1, rots, scales


def reproduce_experiments(video_names, sticker_locations, args):
    """
    reproduces original experiments reported in manuscript, results are printed to log or plotted where applicable
    :param r_matrix: see caller
    :param s_matrix: see caller
    :param video_names: see caller
    :param args: see caller
    :return: -
    """
    # do_network_robustness_test(sticker_locations, args)
    r_matrix, s_matrix = predict.predict_rigid_transform(sticker_locations, None, args)
    # grid_search_sensor_names, grid_search_xyz, grid_search_rots, grid_search_scales = do_parameter_grid_search_experiment(args)
    # do_MNI_sensitivity_experiment(args.template)
    # do_digi_error_experiment()
    dig_ses1, dig_ses2, all_digi_sessions = do_dig2dig_experiment(args.template, args.ground_truth, save_results=False)
    # do_opt2dig_experiment(dig_ses1, dig_ses2, grid_search_xyz, grid_search_rots, grid_search_scales, video_names)
    # vid_ses1, vid_ses2 = do_vid2vid_project_afterMNI_experiment(args.template, video_names, r_matrix, s_matrix)
    vid_ses1, vid_ses2 = do_vid2vid_experiment(args, video_names, r_matrix, s_matrix, force_project=False, save_results=False)
    dig_intra, vid_intra, inter, per_landmark = do_vid2dig_experiment(dig_ses1, dig_ses2, vid_ses1, vid_ses2)
    do_brain_error_visualization_experiment(vid_ses1, per_landmark)
    do_histogram_experiment(dig_ses1, dig_ses2, vid_ses1, vid_ses2)
    do_skull_size_experiment(dig_intra, vid_intra, inter, args)
    do_shift_experiment(r_matrix, all_digi_sessions, args)
    # do_old_experiment(r_matrix, s_matrix, video_names, args)


def do_brain_error_visualization_experiment(vid_ses1, per_patient_landmarks):
    """
    visualizes errors spatially on average MNI brain
    :param vid_ses1: all first video sessions
    :param per_landmark: np array: subjects x landmarks
    landmark-per-patient inter-method error (vid session1 and digitizer sessions)
    :return:
    """
    XYZ = MNI_torch.load_raw_MNI_data("resource" + "/MNI_templates/xyzallBEM.npy", "brain", resource_folder="resource")
    import pptk
    v = pptk.viewer(XYZ)
    v.color_map('hot', scale=[0, 1])
    # color = (XYZ[:, 0] - np.min(XYZ[:, 0])) / (np.max(XYZ[:, 0]) - np.min(XYZ[:, 0]))
    colors = np.empty((len(vid_ses1[1]), XYZ.shape[0]))
    for i in range(len(vid_ses1[1])):
        locs = vid_ses1[1][i]
        weights = per_patient_landmarks[i]
        distances = 1 / (np.linalg.norm(XYZ - np.expand_dims(locs, 1), axis=-1) + 1e-5)
        distances /= np.sum(distances, axis=0)
        non_normalized_colors = weights @ distances
        colors[i] = non_normalized_colors
    colors = np.mean(colors, axis=0)
    max_color = np.max(colors)
    min_color = np.min(colors)
    mean_color = np.mean(colors)
    std_color = np.std(colors)
    logging.info("spatial mean, std (video session): {}, {}".format(mean_color, std_color))
    colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))
    v.attributes(colors)

    v.set(show_grid=False, show_axis=False, show_info=False,
          lookat=(0, 0, 0), phi=np.pi, theta=0, r=300, point_size=0.5)
    v.capture('cache/right_view.png')
    v.set(show_grid=False, show_axis=False, show_info=False,
          lookat=(0, 0, 0), phi=0, theta=0, r=300, point_size=0.5)
    v.capture('cache/left_view.png')
    v.set(show_grid=False, show_axis=False, show_info=False,
          lookat=(0, 0, 0), phi=0, theta=np.pi / 2, r=300, point_size=0.5)
    v.capture('cache/top_view.png')
    v.set(show_grid=False, show_axis=False, show_info=False,
          lookat=(0, 0, 0), phi=+np.pi / 2, theta=0, r=300, point_size=0.5)
    v.capture('cache/front_view.png')
    v.set(show_grid=False, show_axis=False, show_info=False,
          lookat=(0, 0, 0), phi=+np.pi / 4, theta=np.pi / 4, r=300, point_size=0.5)
    v.capture('cache/iso1.png')
    v.set(show_grid=False, show_axis=False, show_info=False,
          lookat=(0, 0, 0), phi=3 * np.pi / 4, theta=np.pi / 4, r=300, point_size=0.5)
    v.capture('cache/iso2.png')
    draw.plot_colorbar(min_color, max_color)
    v.close()


def do_skull_size_experiment(dig_intra, vid_intra, inter, args):
    """
    plots the skull sizes for each subject against their intra & inter errors
    :param dig_intra:
    :param vid_intra:
    :param inter:
    :param args:
    :return:
    """
    experiment_folder_path = args.ground_truth
    experiment_file_list = []
    skull_radii = []
    for exp_file in sorted(experiment_folder_path.glob("*.txt")):
        experiment_file_list.append(exp_file)
    for exp_file in experiment_file_list:
        logging.info(exp_file.name)
        file_names, file_data, file_format, skull = read_template_file(exp_file)
        skull_sizes = []
        for i, session in enumerate(zip(file_names, file_data)):
            if not session[0]:
                continue
            names = session[0]
            data = session[1][:, 0, :] - session[1][:, 1, :]  # subtract second sensor
            data = geometry.to_standard_coordinate_system(names, data)
            skull_sizes.append([np.linalg.norm(data[(names.index('lpa'))] - data[(names.index('rpa'))]),
                                np.linalg.norm(data[(names.index('fpz'))] - data[(names.index('o1'))])])
        skull_radii.append(np.mean(np.array(skull_sizes), axis=0) / 2)
    skull_radii = np.array(skull_radii)
    circumferences = 2 * np.pi * np.sqrt((skull_radii[:, 0] ** 2 + skull_radii[:, 1] ** 2) / 2)
    draw.plot_skull_vs_error(circumferences, dig_intra, vid_intra, inter)


def do_shift_experiment(r_matrix, all_digi_sessions, args):
    """
    finds corrolation between rotational shifts as detected by the digitzier and our method
    :param r_matrix:
    :param all_digi_sessions:
    :return:
    """
    # find video rotations
    vid_rots = [[], [], []]
    for i, r in enumerate(r_matrix):
        vid_rot = R.from_matrix(r)
        vid_rot_euler = vid_rot.as_euler('xyz', degrees=True)
        vid_rots[i % 3].append(vid_rot_euler)
    # find digitizer rotations
    dig_rots = [[], [], []]
    template_names, template_data, template_format, _ = read_template_file(args.template)
    template_data = template_data[0]
    template_names = template_names[0]
    template_data = geometry.to_standard_coordinate_system(template_names, template_data)
    template_mask_data = template_data[(template_names.index('fp1'),
                                        template_names.index('fp2'),
                                        template_names.index('fpz'),
                                        template_names.index('cz'),
                                        template_names.index('o1'),
                                        template_names.index('o2'),
                                        template_names.index('oz'),
                                        template_names.index('f7'),
                                        template_names.index('f8')), :]
    template_face_data = template_data[(template_names.index('lpa'),
                                        template_names.index('rpa'),
                                        template_names.index('lefteye'),
                                        template_names.index('righteye'),
                                        template_names.index('nz'),
                                        template_names.index('nosetip')), :]
    for i, session in enumerate(all_digi_sessions):
        for j, subject in enumerate(session):
            try:
                names = subject[0]
                data = subject[1]
                file_mask_data = data[(names.index('fp1'),
                                       names.index('fp2'),
                                       names.index('fpz'),
                                       names.index('cz'),
                                       names.index('o1'),
                                       names.index('o2'),
                                       names.index('oz'),
                                       names.index('f7'),
                                       names.index('f8')), :]
                file_face_data = data[(names.index('lpa'),
                                       names.index('rpa'),
                                       names.index('lefteye'),
                                       names.index('righteye'),
                                       names.index('nz'),
                                       names.index('nosetip')), :]
                # align faces as intermediate step
                ret_R, ret_t = geometry.rigid_transform_3d_nparray(template_face_data, file_face_data)
                aligned_template_mask = (ret_R @ template_mask_data.T).T + ret_t
                # find mask transform
                ret_R, ret_t = geometry.rigid_transform_3d_nparray(aligned_template_mask, file_mask_data)
                gt_rot_m = R.from_matrix(ret_R)
                gt_rot_e = gt_rot_m.as_euler('xyz', degrees=True)
                logging.info("Digitizer Euler angels: " + str(gt_rot_e))
                dig_rots[i].append(gt_rot_e)
            except AttributeError:
                dig_rots[i].append(None)
    ########
    dig_rots = [x[:-1] for x in dig_rots]
    vid_rots = [x[:-1] for x in vid_rots]
    digi2digi_shift = np.array(
        [z - (x + y) / 2 for x, y, z in zip(dig_rots[0], dig_rots[1], dig_rots[2])])
    vid2vid_shift = np.array(
        [z - (x + y) / 2 for x, y, z in zip(vid_rots[0], vid_rots[1], vid_rots[2])])

    # correlate the shifts
    x_corr_new, px = pearsonr(digi2digi_shift[:, 0], vid2vid_shift[:, 0])
    y_corr_new, py = pearsonr(digi2digi_shift[:, 1], vid2vid_shift[:, 1])
    z_corr_new, pz = pearsonr(digi2digi_shift[:, 2], vid2vid_shift[:, 2])

    logging.info(
        "Shift correlation (x, px, y, py, z, pz):"
        "{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(x_corr_new, px, y_corr_new, py, z_corr_new, pz))


def do_digi_error_experiment():
    """
    reproduces experiment where various number of optodes were measured with digitizer
    tries to find relationship between number of optodes and error received
    :return:
    """
    try:
        data = read_digitizer_multi_noptodes_experiment_file("resource/digi_error_exp.txt")
    except FileNotFoundError:
        logging.info("cant perform digitizer error experiment, raw data not found.")
        return
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


def do_dig2dig_experiment(template_path, experiment_folder, experiment_filter=None, verbose=True, save_results=True):
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
    sessions = [[], [], []]

    origin_names = ["lpa", "rpa", "nz", "cz", "lefteye", "righteye", "nosetip"]
    others_names = ["fp1", "fp2", "fpz", "o1", "o2", "oz", "f7", "f8"]
    full_names = origin_names + others_names
    if experiment_folder.is_dir():
        for exp_file in sorted(experiment_folder.glob("**/*.txt")):
            experiment_file_list.append(exp_file)
    else:
        experiment_file_list.append(experiment_folder)
    for exp_file in experiment_file_list:
        if verbose:
            logging.info(exp_file.name)
        file_names, file_data, file_format, skull = read_template_file(exp_file, input_file_format="telaviv2")
        for i, session in enumerate(zip(file_names, file_data)):
            try:
                names = session[0]
                data = session[1][:, 0, :] - session[1][:, 1, :]  # subtracts second sensor
                # data = geometry.to_standard_coordinate_system(names, data)
                origin_selector = tuple([names.index(x) for x in origin_names])
                origin = data[origin_selector, :]
                others_selector = tuple([names.index(x) for x in others_names])
                others = data[others_selector, :]
                # others_selector_template = tuple([template_names.index(x) for x in others])
                # W = geometry.affine_transform_3d_nparray(template_data[others_selector_template, :],
                #                                                                  subject_secific_others)
                # others_selector_template = tuple([template_names.index(x) for x in others_template])
                # others_data = template_data[others_selector_template, :]
                # new_others_data = np.c_[others_data, np.ones(len(others_data))]
                # new_others_data_transformed = np.matmul(new_others_data, W)[:,:-1]
                # subject_secific_others = np.vstack((subject_secific_others, new_others_data_transformed))
                sessions[i].append([full_names, np.vstack((origin, others))])
            except IndexError:
                sessions[i].append([None, None])
    errors_before_projection = [np.mean(np.linalg.norm(x[1] - y[1],axis=-1)) for x, y in zip(sessions[0], sessions[1])]
    cached_result_ses1 = "cache/session1_digi_MNI"
    cached_result_ses2 = "cache/session2_digi_MNI"
    if not Path(cached_result_ses1 + ".npy").is_file() or not Path(cached_result_ses2 + ".npy").is_file():
        digi_projected_ses1 = geometry.project_sensors_to_MNI(sessions[0], origin_names)
        digi_ss_data_ses1 = np.array([x[1] for x in digi_projected_ses1])
        digi_projected_ses2 = geometry.project_sensors_to_MNI(sessions[1], origin_names)
        digi_ss_data_ses2 = np.array([x[1] for x in digi_projected_ses2])
        if save_results:
            np.save(cached_result_ses1, digi_ss_data_ses1)
            np.save(cached_result_ses2, digi_ss_data_ses2)
    else:
        digi_ss_data_ses1 = np.load(cached_result_ses1 + ".npy", allow_pickle=True)
        digi_ss_data_ses2 = np.load(cached_result_ses2 + ".npy", allow_pickle=True)
    if experiment_filter:
        id = [i for (i, x) in enumerate(experiment_file_list) if experiment_filter in str(x)][0]
        digi_projected_ses1_others = digi_ss_data_ses1[id:id+1, full_names.index(others_names[0]):, :]
        digi_projected_ses2_others = digi_ss_data_ses2[id:id+1, full_names.index(others_names[0]):, :]
    else:
        digi_projected_ses1_others = digi_ss_data_ses1[:, full_names.index(others_names[0]):, :]
        digi_projected_ses2_others = digi_ss_data_ses2[:, full_names.index(others_names[0]):, :]

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
    for i in range(len(digi_projected_ses2_others)):
        rmse_error = geometry.get_rmse(digi_projected_ses1_others[i], digi_projected_ses2_others[i])
        errors.append(rmse_error)
    rmse_error_mean = np.mean(np.array(errors))
    rmse_error_std = np.std(np.array(errors))
    if verbose:
        logging.info("dig2dig mean, std rmse (after MNI projection): {:.3f}, {:.3f}".format(rmse_error_mean,
                                                                                            rmse_error_std))

    # from draw import visualize_2_pc
    # visualize_2_pc(points_blue=digi_projected_ses1_others[0], points_red=digi_projected_ses2_others[0])
    return [others_names, digi_projected_ses1_others],\
           [others_names, digi_projected_ses2_others],\
           sessions


def do_vid2vid_project_afterMNI_experiment(template_path, video_names, r_matrices, s_matrices):
    """
    experiement to compare intra-method error between video sessions
    mni projection is done before network transformation is applied
    :param template_path:
    :param video_names:
    :param r_matrices:
    :param s_matrices:
    :return:
    """
    names, data, format, _ = read_template_file(template_path)
    names = names[0]
    data = data[0]
    data = geometry.to_standard_coordinate_system(names, data)
    output_others_names = ["fp1", "fp2", "fpz", "o1", "o2", "oz", "f7", "f8"]
    others_names = output_others_names + names[names.index(0):]
    origin_names = ["lpa", "rpa", "nz", "cz"]
    full_names = origin_names + others_names
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
        sessions[session_number].append([full_names, np.vstack((origin, others))])
        rots[session_number].append(rot_mat)
    cached_result_ses1 = "cache/session1_vid_MNI_transa"
    cached_result_ses2 = "cache/session2_vid_MNI_transa"
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
    vid_projected_ses1_others = vid_ss_data_ses1[:, full_names.index(output_others_names[0]):, :]
    vid_projected_ses2_others = vid_ss_data_ses2[:, full_names.index(output_others_names[0]):, :]
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


def do_vid2vid_experiment(opt, video_names, r_matrices, s_matrices, force_project=False, save_results=True):
    """
    experiement to compare intra-method error between video sessions
    applies transform from network before MNI projection
    :param template_path:
    :param video_names:
    :param r_matrices:
    :param s_matrices:
    :param digi_sessions: pass this to use anchors from digitizer
    :return:
    """
    template_path = opt.template
    names, data, format, _ = read_template_file(template_path)
    names = names[0]
    data = data[0]
    data = geometry.to_standard_coordinate_system(names, data)
    output_others_names = ["fp1", "fp2", "fpz", "o1", "o2", "oz", "f7", "f8"]
    others_names = output_others_names + names[names.index(0):]
    origin_names = ["lpa", "rpa", "nz", "cz"]
    full_names = origin_names + others_names
    cached_result_ses1 = "cache/session1_vid_MNI_transb"
    cached_result_ses2 = "cache/session2_vid_MNI_transb"
    if not Path(cached_result_ses1 + ".npy").is_file() or not Path(cached_result_ses2 + ".npy").is_file() or force_project:
        if True: #opt.gpu_id == "cpu":
            sessions = [[], []]
            for i, (rot_mat, scale_mat, vid) in enumerate(zip(r_matrices, s_matrices, video_names)):
                subject_id, _, session_number = vid.split("_")
                session_number = int(session_number.split(".")[0]) - 1
                if session_number == 2:
                    continue
                origin_selector = tuple([names.index(x) for x in origin_names])
                origin = data[origin_selector, :]
                others_selector = tuple([names.index(x) for x in others_names])
                others = data[others_selector, :]
                others = (rot_mat @ (
                            scale_mat @ others.T)).T  # apply transform before MNI projection, but before digi transform if applicaple
                sessions[session_number].append([full_names, np.vstack((origin, others))])
            vid_projected_ses1 = geometry.project_sensors_to_MNI(sessions[0], origin_names)
            anchors_and_sensors_ses1 = np.array([x[1] for x in vid_projected_ses1], dtype=np.float)
            vid_projected_ses2 = geometry.project_sensors_to_MNI(sessions[1], origin_names)
            anchors_and_sensors_ses2 = np.array([x[1] for x in vid_projected_ses2], dtype=np.float)
        else:
            origin_selector = tuple([names.index(x) for x in origin_names])
            others_selector = tuple([names.index(x) for x in others_names])
            anchors = data[origin_selector, :]
            sensors = data[others_selector, :]
            transformed_sensors = (np.array(r_matrices) @ (np.array(s_matrices) @ sensors.T)).transpose(0, 2, 1)
            sorted_anchors, indices = geometry.sort_anchors(np.array(origin_names), anchors)
            device = opt.gpu_id
            origin_xyz_torch = torch.from_numpy(sorted_anchors).float().to(device)
            sensors_xyz_torch = torch.from_numpy(transformed_sensors).float().to(device)
            selected_indices_torch = torch.from_numpy(indices).to(device)
            ses1_torch_mni, _, _ = MNI_torch.torch_project_non_differentiable(origin_xyz_torch,
                                                                              sensors_xyz_torch[::3, :, :],
                                                                              selected_indices_torch,
                                                                              output_errors=False)
            ses2_torch_mni, _, _ = MNI_torch.torch_project_non_differentiable(origin_xyz_torch,
                                                                              sensors_xyz_torch[1::3, :, :],
                                                                              selected_indices_torch,
                                                                              output_errors=False)
            np_mni_ses1 = ses1_torch_mni.cpu().numpy()
            np_mni_ses2 = ses2_torch_mni.cpu().numpy()

            # stack anchors and projected sensors
            anchors_and_sensors_ses1 = np.concatenate((np.tile(np.expand_dims(anchors, axis=0), (len(ses1_torch_mni), 1, 1)), np_mni_ses1), axis=1)
            anchors_and_sensors_ses2 = np.concatenate((np.tile(np.expand_dims(anchors, axis=0), (len(ses2_torch_mni), 1, 1)), np_mni_ses2), axis=1)
        if save_results:
            np.save(cached_result_ses1, anchors_and_sensors_ses1)
            np.save(cached_result_ses2, anchors_and_sensors_ses2)
    else:
        anchors_and_sensors_ses1 = np.load(cached_result_ses1 + ".npy", allow_pickle=True)
        anchors_and_sensors_ses2 = np.load(cached_result_ses2 + ".npy", allow_pickle=True)
    vid_projected_ses1_others = anchors_and_sensors_ses1[:, full_names.index(output_others_names[0]):, :]
    vid_projected_ses2_others = anchors_and_sensors_ses2[:, full_names.index(output_others_names[0]):, :]
    errors = []
    for i in range(len(vid_projected_ses1_others)):
        rmse_error = geometry.get_rmse(vid_projected_ses1_others[i], vid_projected_ses2_others[i])
        errors.append(rmse_error)
    rmse_error_mean = np.mean(np.array(errors))
    rmse_error_std = np.std(np.array(errors))
    logging.info("vid2vid mean, std rmse (after MNI projection): {:.3f}, {:.3f}".format(rmse_error_mean,
                                                                                        rmse_error_std))
    output_selector = tuple([full_names.index(x) for x in output_others_names])
    return [output_others_names, anchors_and_sensors_ses1[:, output_selector, :]],\
           [output_others_names, anchors_and_sensors_ses2[:, output_selector, :]]


def do_vid2dig_experiment(digi_ses1, digi_ses2, vid_ses1, vid_ses2):
    """
    reports inter method errors (and t-tests, which are not that relevant)
    :param digi_ses1:
    :param digi_ses2:
    :param vid_ses1:
    :param vid_ses2:
    :return: intra method reliability for digitizer, for video, and inter method validity.
    """
    inter_method_rmse1 = [geometry.get_rmse(x, y) for x, y in zip(digi_ses1[1], vid_ses1[1])]
    inter_method_rmse2 = [geometry.get_rmse(x, y) for x, y in zip(digi_ses1[1], vid_ses2[1])]
    inter_method_rmse3 = [geometry.get_rmse(x, y) for x, y in zip(digi_ses2[1], vid_ses1[1])]
    inter_method_rmse4 = [geometry.get_rmse(x, y) for x, y in zip(digi_ses2[1], vid_ses2[1])]
    inter_method_rmse_avg = np.mean([inter_method_rmse1, inter_method_rmse2, inter_method_rmse3, inter_method_rmse4])
    inter_method_rmse_std = np.std([inter_method_rmse1, inter_method_rmse2, inter_method_rmse3, inter_method_rmse4])

    # for visualization, find best error per landmark on video session1
    vid_combine = vid_ses1[1]
    dig_combine = (digi_ses1[1] + digi_ses2[1]) / 2
    landmarks_errors = np.linalg.norm((dig_combine - vid_combine).astype(np.float), axis=-1)

    logging.info("dig2vid mean, std rmse (after MNI projection): {:.3f}, {:.3f}".format(inter_method_rmse_avg,
                                                                                        inter_method_rmse_std))

    digi_intra_method_rmse = [geometry.get_rmse(x, y) for x, y in zip(digi_ses1[1], digi_ses2[1])]
    vid_intra_method_rmse = [geometry.get_rmse(x, y) for x, y in zip(vid_ses1[1], vid_ses2[1])]
    # calc t-test statistics
    a_vid = np.array(vid_intra_method_rmse)
    a_dig = np.array(digi_intra_method_rmse)
    a_inter = np.mean(np.array([inter_method_rmse1, inter_method_rmse2, inter_method_rmse3, inter_method_rmse4]),
                      axis=0)
    # probably should be ttest_ind
    t1, p1 = ttest_rel(a_vid, a_dig)
    logging.info("t-test results intra (t, p): {:.3f}, {:.3f}".format(t1, p1))
    # probably should be ttest_ind
    t2, p2 = ttest_rel(a_dig, a_inter)
    logging.info("t-test results inter (t, p): {:.3f}, {:.3f}".format(t2, p2))
    return a_dig, a_vid, a_inter, landmarks_errors


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
    template_face_data = template_data[(template_names.index('lpa'),
                                        template_names.index('rpa'),
                                        template_names.index('lefteye'),
                                        template_names.index('righteye'),
                                        template_names.index('nz'),
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
            data = session[1][:, 0, :] - session[1][:, 1, :]  # subtract second sensor
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
            file_face_data = data[(names.index('lpa'),
                                   names.index('rpa'),
                                   names.index('lefteye'),
                                   names.index('righteye'),
                                   names.index('nz'),
                                   names.index('nosetip')), :]
            # align faces as intermediate step
            ret_R, ret_t = geometry.rigid_transform_3d_nparray(template_face_data, file_face_data)
            aligned_template_mask = (ret_R @ template_mask_data.T).T + ret_t

            if spiral_output_type == "orig":
                subject_secific_origin = data[(names.index('lpa'),
                                               names.index('rpa'),
                                               names.index('nz'),
                                               names.index('cz')), :]
                subject_secific_spiral = file_mask_data
                spiral_output.append([subject_secific_origin, subject_secific_spiral])
            elif spiral_output_type == "spiral_transformed":
                if i == 0:
                    subject_secific_origin = data[(names.index('lpa'),
                                                   names.index('rpa'),
                                                   names.index('nz'),
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
            skull_sizes.append([np.linalg.norm(data[(names.index('lpa'))] - data[(names.index('rpa'))]),
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


def do_MNI_sensitivity_experiment(template_path):
    """
    experiment where noise is added to anchors before MNI projection
    then we compare the output between the noisy and non-noisy projections
    this verfies how crucial are the anchors for the projection.
    :param template_path:
    :return:
    """
    iters = 1
    template_names, template_data, _, _ = read_template_file(template_path)
    template_names = template_names[0]
    template_data = template_data[0]
    template_data = geometry.to_standard_coordinate_system(template_names, template_data)
    origin_names = ["lpa", "rpa", "nz", "cz"]
    others_names = ["fp1", "fp2", "fpz", "o1", "o2", "oz", "f7", "f8"]
    others_template = template_names[template_names.index(0):]
    combined_others = others_names #+ others_template
    full_names = origin_names + combined_others
    origin_selector = tuple([template_names.index(x) for x in origin_names])
    origin = template_data[origin_selector, :]
    others_selector = tuple([template_names.index(x) for x in combined_others])
    others = template_data[others_selector, :]
    session = [full_names, np.vstack((origin, others))]
    session_proj1 = geometry.project_sensors_to_MNI([session], origin_names)
    tot_errors = []
    for i in range(len(origin)):
        errors = []
        for j in range(iters):
            new_origin = origin.copy()
            new_origin[i] += (np.random.rand(3) - 0.5)*0.2
            session = [full_names, np.vstack((new_origin, others))]
            session_proj2 = geometry.project_sensors_to_MNI([session], origin_names)
            rmse = geometry.get_rmse(session_proj1[0][1][len(origin_names):], session_proj2[0][1][len(origin_names):])
            errors.append(rmse)
        tot_errors.append(np.mean(np.array(errors)))
    logging.info(["{}->{:.3f}mm".format(x, y) for x, y in zip(origin_names, tot_errors)])


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
    digi_names = ["lpa", "rpa", "nz", "cz"] + list_extension
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
            data_origin = data[:names.index(0), :]  # non numbered optodes are not coregistered
            data_optodes = data[names.index(0):, :]  # selects optodes for applying coregistration
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
