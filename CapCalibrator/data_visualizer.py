import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np
import utils

def load_db(db_path, format="pickle"):
    if format == "pickle":
        f = open(db_path, 'rb')
        db = pickle.load(f)
        f.close()
    else:
        if format == "json":
            number_of_samples = 20
            skip_files = 0
            count = 0
            db = {}
            count = 0
            for i, file in enumerate(db_path.glob("*.json")):
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
    db = load_db(db_path, my_format)
    if filter is not None and my_format=="pickle":
        new_db = {}
        for file in filter:
            new_db[file] = db.pop(file, None)
        db = new_db
    # db = fix_db(db)
    # save_db(db, db_path)
    for key in db.keys():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data = db[key]["data"][0]
        s_linear = [n for n in range(len(data))]
        if format=="pickle":
            c = ['b', 'b', 'b', 'r', 'r', 'r', 'g']
        else:
            c = ['b', 'b', 'b', 'g', 'r', 'r', 'r']
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


filter_files = ["GX011577.MP4", "GX011578.MP4", "GX011579.MP4", "GX011580.MP4",
                "GX011581.MP4", "GX011582.MP4", "GX011572.MP4", "GX011573.MP4",
                "GX011574.MP4", "GX011575.MP4", "GX011576.MP4", "GX011566.MP4",
                "GX011567.MP4", "GX011568.MP4", "GX011569.MP4", "GX011570.MP4"]
# db_path = Path("data", "full_db.pickle")
db_path = Path("E:/Src/CapCalibrator/DataSynth/captures")
visualize_data(db_path, filter=filter_files)
