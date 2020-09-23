import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "5";
import keras
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np
import utils
import file_io
import video_annotator
import draw


def visualize_network_performance(model_name, root_dir):
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
    best_weight_location = Path.joinpath(model_dir, "{}_best_weights.h5".format(model_name))
    model = keras.models.load_model(str(best_weight_location))
    gt_pred = model.predict(np.expand_dims(A, axis=0))
    y_predict_special = model.predict(np.expand_dims(x_train[min_index], axis=0))
    print("gt pred: ", gt_pred, gt[vid_name]["label"])
    print("synth pred: ", y_predict_special, y_train[min_index])
    filter = [vid_name]
    draw.visualize_annotated_data(db_path, filter)
    filter = ["image_{:05d}.json".format(min_index)]
    draw.visualize_annotated_data(synth_data_dir, filter)
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


def save_db(db, db_path):
    f = open(db_path, 'wb')
    pickle.dump(db, f)
    f.close()


def fix_db(db):
    for key in db.keys():
        if db[key]["data"].shape[1] == 11:
            db[key]["data"] = np.delete(db[key]["data"], 10, axis=1)
            db[key]["frame_indices"] = db[key]["frame_indices"][:-1]
    return db

# def fix_special_db(db):
#     fresh_db = {}
#     special_db_path = Path.joinpath(Path("data"), "full_db_special.pickle")
#     for key in db.keys():
#         fresh_db.setdefault(key[:-3], []).append({"data": db[key]["data"],
#                                                   "label": db[key]["label"],
#                                                   "frame_indices": db[key]["frame_indices"]})
#     save_full_db(fresh_db, special_db_path)
#     print("done")

# def fix_normal_db(db):
#     fresh_db = {}
#     db_path = Path.joinpath(Path("data"), "full_db.pickle")
#     for key in db.keys():
#         fresh_db.setdefault(key, []).append({"data": db[key]["data"],
#                                                   "label": db[key]["label"],
#                                                   "frame_indices": db[key]["frame_indices"]})
#     save_full_db(fresh_db, db_path)
#     print("done")


def doSFM(video_path):
    video_name = "GX011635.MP4"
    # full_db = video_annotator.load_full_db()
    # frames_indices = full_db[video_name]
    frames, indices = video_annotator.process_video(video_path,
                                                    dump_frames=False,
                                                    starting_frame=0,
                                                    force_reselect=False,
                                                    frame_indices=None,
                                                    v=0)
    for i, frame in enumerate(frames):
        # im = Image.fromarray(frame)
        frame.save("plots/sfm/IM{:03d}.jpg".format(i))