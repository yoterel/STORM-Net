# import tensorflow as tf
import tf_file_io
import utils
from pathlib import Path
import cv2
import pickle
from scipy.spatial.transform import Rotation as R
import numpy as np
import logging
import data_augmentations
import torch
import torch_src.torch_model as torch_model
import torch_src.torch_data as torch_data


class Options:
    def __init__(self, device):
        self.network_input_size = 10
        self.template = Path("../example_models/example_model.txt")
        self.architecture = "2dconv"
        self.loss = "l2"
        self.device = device
        self.scale_faces = None
        self.network_output_size = 3
        if self.scale_faces:
            self.network_output_size += len(self.scale_faces)


def is_using_gpu():
    return torch.cuda.is_available()


def predict_rigid_transform(sticker_locations, preloaded_model, args):
    """
    predicts rigid transformation of cap object using 2d sticker locations
    :param sticker_locations: a batch of 2d array of sticker locations
    :param preloaded_model: a pre loaded keras model
    :param args: command line arguments
    :return: rotation and scale matrices list
    """
    # scale to 0-1 for network
    sticker_locations[:, :, 0::2] /= 960
    sticker_locations[:, :, 1::2] /= 540
    # mask facial landmarks for frames that have less than 3 of them
    data_augmentations.mask_facial_landmarks(sticker_locations)
    # center the data
    data_augmentations.center_data(sticker_locations)
    if preloaded_model:
        network = preloaded_model
    else:
        model_full_path = Path(args.storm_net)
        opt = Options(device=args.gpu_id)
        network = torch_model.MyNetwork(opt)
        state_dict = torch.load(model_full_path, map_location=args.gpu_id)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        network.load_state_dict(state_dict)
        network.to(opt.device)
    heat_mapper = torch_data.HeatMap((256, 256), 16, False, args.gpu_id)
    x = torch.from_numpy(sticker_locations).to(args.gpu_id).float()
    x[:, :, 0::2] *= 256
    x[:, :, 1::2] *= 256
    y_predict = torch.empty((len(x), network.opt.network_output_size), dtype=torch.float, device=args.gpu_id)
    for i in range(len(x)):
        heatmap = heat_mapper(x[i].reshape(10, x[i].shape[-1] // 2, 2))
        with torch.no_grad():
            _, pred = network(heatmap.unsqueeze(0))
            y_predict[i] = pred
    y_predict = y_predict.cpu().numpy()
    assert y_predict.shape[-1] == network.opt.network_output_size
    # simulation uses left hand rule (as opposed to scipy rotation that uses right hand rule)
    # notice x is not negated - the positive direction in simulation is flipped.
    rs = []
    sc = []
    for i in range(len(y_predict)):
        logging.info("Storm-Net Parameters Prediction:" + str(y_predict[i].tolist()))
        rot = R.from_euler('xyz', [y_predict[i][0], y_predict[i][1], y_predict[i][2]], degrees=True)
        scale_mat = np.identity(3)
        if y_predict.shape[-1] > 3:
            counter = 0
            if "x" in network.opt.scale_faces:
                xterm = y_predict[i][3 + counter]
                counter += 1
            else:
                xterm = 1.0
            if "y" in network.opt.scale_faces:
                yterm = y_predict[i][3 + counter]
                counter += 1
            else:
                yterm = 1.0
            if "z" in network.opt.scale_faces:
                zterm = y_predict[i][3 + counter]
            else:
                zterm = 1.0
            scale_mat[0, 0] = xterm  # xscale
            scale_mat[1, 1] = yterm  # yscale
            scale_mat[2, 2] = zterm  # zscale
        rotation_mat = rot.as_matrix()
        rs.append(rotation_mat)
        sc.append(scale_mat)
    return rs, sc


def get_facial_landmarks(frames):
    """
    predicts location of center of eyes and nose tip in a set of images
    :param frames: the images to predict the landmarks on
    :return: a 2d numpy array containing x, y coordinates of required landmarks for each frame
    """
    import dlib
    model_path = Path("models", "shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(model_path))
    landmarks_list = []
    for j, frame in enumerate(frames):
        my_landmarks = np.tile(np.array([0, 540]), 3)
        img_data = np.array(frame)
        rects = detector(img_data, 0)
        if len(rects) > 1:
            logging.info("Warning: found more than 1 face in frame: " + str(j))
            rects = [max(rects, key=lambda x: x.top())]  # select bottom most
        if rects:
            if rects[0].top() < 100:
                rects = []
        for (i, rect) in enumerate(rects):
            # Make the prediction and transform it to numpy array
            landmarks = predictor(img_data, rect)
            landmarks = utils.shape_to_np(landmarks)
            # left eye, nose, right eye
            my_landmarks = np.array([np.mean(landmarks[42:48, 0]),
                                    np.mean(landmarks[42:48, 1]),
                                    np.mean(landmarks[30, 0]),
                                    np.mean(landmarks[30, 1]),
                                    np.mean(landmarks[36:42, 0]),
                                    np.mean(landmarks[36:42, 1])])
        landmarks_list.append(my_landmarks)
    np_kp = np.array(landmarks_list)
    np_kp[:, 1::2] = 540 - np_kp[:, 1::2]
    return np_kp


def get_blob_keypoints(mask, max_key_points, facial_landmarks=False):
    """
    finds blobs in a binary mask image
    :param mask: the mask image
    :param max_key_points: maximum key points allowed
    :param facial_landmarks: if true, will slightly dilate and erode image
    :return: numpy array of locations of center of blobs obeying some heuristics
    """
    # if facial_landmarks:
    #     kernel = np.ones((5, 5), np.uint8)
    #     mask = cv2.dilate(mask, kernel, iterations=2)
    #     # plt.imshow(mask)
    #     # plt.show()
    #     mask = cv2.erode(mask, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    keypoints = []
    for i, contour in enumerate(contours):
        M = cv2.moments(contour)
        if M['m00']:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            area = cv2.contourArea(contour)
            keypoints.append((cx, cy, area, i))
    keypoints.sort(key=lambda tup: tup[2], reverse=True)
    keypoints = [x for x in keypoints if x[2] > 200]  # area filter
    keypoints = [x for x in keypoints if 300 < x[0] < 700]  # location filter
    if len(keypoints) > max_key_points:  # reduce blob numbers
        keypoints = keypoints[0:max_key_points]
    keypoints = np.array([np.array([x[0], x[1]]) for x in keypoints])
    return keypoints


def get_sticker_locations(frames, preloaded_model, graph, args):
    """
    predicts green sticker locations in a set of frames
    :param frames: the frames to process
    :param preloaded_model: a preloaded keras model to use
    :param graph: default tf graph
    :param args: cmd line arguments
    :return: locations of stickers in all frames as a 2d numpy array
    """
    # if preloaded_model:
    #     my_model = preloaded_model
    # else:
    model_full_path = Path(args.u_net)
    my_model, graph = tf_file_io.load_semantic_seg_model(str(model_full_path))
    imgs_list = []
    for image in frames:
        img_data = np.array(image.resize((1024, 512)))  # our unet only accepts powers of 2 image sizes
        imgs_list.append(img_data)
    imgs_np = np.array(imgs_list)
    x = np.asarray(imgs_np, dtype=np.float32) / 255
    y_pred_list = []
    for i in range(x.shape[0]):
        to_predict = np.expand_dims(x[i], axis=0)
        with graph.as_default():
            y_pred = my_model.predict(to_predict)
        y_pred_list.append(y_pred)
    y_pred_np = np.array(y_pred_list)
    y_pred_np = np.squeeze(y_pred_np)
    threshold, upper, lower = 0.5, 1, 0
    y_pred_np = np.where(y_pred_np > threshold, upper, lower)
    y_pred_np = y_pred_np.astype(np.uint8)
    key_points_list = []
    logging.info("Filtering & extracting blobs.")
    for i in range(len(y_pred_np)):
        key_points = get_blob_keypoints(y_pred_np[i], 4, False)
        key_points_list.append(key_points.flatten())
    # pad with zeros until we reach 2x4 numbers
    for i in range(len(key_points_list)):
        padding = np.zeros(8)
        key_points_list[i][1::2] = 512 - key_points_list[i][1::2]
        padding[0:key_points_list[i].shape[0]] = key_points_list[i]
        key_points_list[i] = padding
    kp_np = np.array(key_points_list)
    kp_np[:, 0::2] *= 960 / 1024
    kp_np[:, 1::2] *= 540 / 512
    return kp_np


def predict_keypoints_locations(frames, args, vid_hash="", is_puppet=False, save_intermed=True, preloaded_model=None, graph=None):
    """
    predicts all requried keypoints (stickers & facial landmarks) locations from frames.
    :param frames: the frames to process
    :param args: cmd line arguments
    :param vid_hash: a pickle file will be loaded if exists to save time (from previous runs)
    :param is_puppet: legacy landmark detector, to be removed.
    :param save_intermed: if true intermediate products will be saved to disk
    :param preloaded_model: a preloaded keras model to be used for prediction
    :param graph: tf graph
    :return: locations as a 2d numpy array in the order "Left Eye, Nose, Right Eye, x1, x2, x3, x4" where x is an arbitrary sticker
    """
    pickle_path = Path("cache", vid_hash+"_preds.pickle")
    if pickle_path.is_file():
        logging.info("Loading key points from: " + str(pickle_path))
        f = open(pickle_path, 'rb')
        key_points = pickle.load(f)
    else:
        logging.info("Detecting facial key points.")
        facial_keypoints = get_facial_landmarks(frames)
        logging.info("Detecting sticker key points.")
        sticker_keypoints = get_sticker_locations(frames, preloaded_model, graph, args)
        key_points = np.concatenate((facial_keypoints, sticker_keypoints), axis=1)
        key_points = np.expand_dims(key_points, axis=0)
        if save_intermed:
            cache_path = Path("cache")
            cache_path.mkdir(exist_ok=True)
            f = open(pickle_path, 'wb')
            pickle.dump(key_points, f)
            f.close()
    return key_points
