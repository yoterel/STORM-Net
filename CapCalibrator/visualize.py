import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "4";
import keras
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np
import utils


def visualize_network_performance(model_name, root_dir):
    #############################################################
    db_path = Path("data", "full_db.pickle")
    gt = load_gt_db(db_path)
    vid_name = "GX011578.MP4"
    synth_data_dir = Path.joinpath(root_dir, "captures_special")
    pickle_file_path = Path.joinpath(synth_data_dir, "data.pickle")
    if not pickle_file_path.is_file():
        X, Y = utils.load_db(synth_data_dir)
        x_train, x_val, y_train, y_val, x_test, y_test = utils.split_data(X, Y)
        utils.serialize_data(pickle_file_path, x_train, x_val, y_train, y_val, x_test, y_test)
    else:
        x_train, x_val, y_train, y_val, x_test, y_test = utils.deserialize_data(pickle_file_path)
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
    visualize_data(db_path, filter)
    filter = ["image_{:05d}.json".format(min_index)]
    visualize_data(synth_data_dir, filter)
#################################################################
    pickle_file_path = Path.joinpath(data_dir, "data.pickle")
    x_train, x_val, y_train, y_val, x_test, y_test = utils.deserialize_data(pickle_file_path)
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


def load_gt_db(db_path, format="pickle", filter=None):
    db = {}
    if format == "pickle":
        f = open(db_path, 'rb')
        db = pickle.load(f)
        f.close()
        if filter is not None:
            new_db = {}
            for file in filter:
                new_db[file] = db.pop(file, None)
            db = new_db
    else:
        if format == "json":
            number_of_samples = 15
            skip_files = 0
            count = 0
            for i, file in enumerate(db_path.glob("*.json")):
                if filter:
                    if file.name in filter:
                        x, y = utils.extract_session_data(file, use_scale=False)
                        x = np.expand_dims(x, axis=0)
                        x[:, :, 0::2] *= 960
                        x[:, :, 1::2] *= 540
                        db[file.name] = {"data": x, "label": y}
                else:
                    if i < skip_files:
                        continue
                    x, y = utils.extract_session_data(file, use_scale=False)
                    if x is not None:
                        x = np.expand_dims(x, axis=0)
                        x[:, :, 0::2] *= 960
                        x[:, :, 1::2] *= 540
                        db[file.name] = {"data": x, "label": y}
                        count += 1
                    if count >= number_of_samples:
                        break
    return db


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


def visualize_data(db_path, filter=None):
    my_format = "pickle" if db_path.suffix == ".pickle" else "json"
    db = load_gt_db(db_path, my_format, filter)
    # db = fix_db(db)
    # save_db(db, db_path)
    for key in db.keys():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data = db[key]["data"][0]
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
        ax.set_xlim([0, 960])
        ax.set_ylim([0, 540])
        # plt.show()
        plt.savefig(Path("plots", "visualize_data", key+".png"))


def visualize_pc(points_blue, names_blue, points_red=None, names_red=None, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if points_red is not None:
        for i in range(len(points_blue)):  # plot each point + it's index as text above
            ax.scatter(points_blue[i, 0], points_blue[i, 1], points_blue[i, 2], color='b')
            ax.text(points_blue[i, 0],
                    points_blue[i, 1],
                    points_blue[i, 2],
                    '%s' % (names_blue[i]),
                    size=20,
                    zorder=1,
                    color='k')
        for i in range(len(points_red)):
            ax.scatter(points_red[i, 0], points_red[i, 1], points_red[i, 2], color='r')
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


if __name__ == "__main__":
    # model_name = 'scene3_batch16_lr1e4_supershuffle_noise6'
    # root_dir = Path("/disk1/yotam/capnet")
    #
    # visualize_network_performance(model_name, root_dir)

    # filter_files = ["GX011577.MP4", "GX011578.MP4", "GX011579.MP4", "GX011580.MP4",
    #                 "GX011581.MP4", "GX011582.MP4", "GX011572.MP4", "GX011573.MP4",
    #                 "GX011574.MP4", "GX011575.MP4", "GX011576.MP4", "GX011566.MP4",
    #                 "GX011567.MP4", "GX011568.MP4", "GX011569.MP4", "GX011570.MP4"]
    # db_path = Path("data", "full_db.pickle")
    db_path = Path("E:/Src/CapCalibrator/DataSynth/captures")
    visualize_data(db_path, filter=None)