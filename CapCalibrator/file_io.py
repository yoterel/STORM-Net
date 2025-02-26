import numpy as np
import subprocess
import pickle
import json
import logging
import shutil
from pathlib import Path
import re
from tkinter import filedialog


def select_from_filesystem(isdir, exists, initial_dir, title):
        """
        selects a filesystem object and returns its path
        :param isdir: is object to select a dir? (else assumes it is a file)
        :param exists: does the filesystem object exist? ignored if isdir is true
        :param initial_dir: the initial popup dialog dir
        :param title: the title of the popup dialog
        :return: the path to the object or None if failure occurs
        """
        if isdir:
            folder = filedialog.askdirectory(initialdir=initial_dir, title=title)
            if not folder:
                return None
            else:
                return Path(folder)
        else:
            if exists:
                file = filedialog.askopenfilename(initialdir=initial_dir, title=title)
                if not file:
                    return None
                else:
                    return Path(file)
            else:
                file = filedialog.asksaveasfile(initialdir=initial_dir, title=title)
                if not file:
                    return None
                else:
                    # todo: okay to forget handle?
                    return Path(file.name)

def pairwise(iterable):
    """
    turns an iterable into a pairwise iterable
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    :param iterable:
    :return:
    """
    a = iter(iterable)
    return zip(a, a)


def read_template_file(template_path, input_file_format=None):
    """
    reads a template file in telaviv format ("sensor x y z rx ry rz") or in princeton format ("name x y z")
    multiple sessions in same file are assumed to be delimited by a line "*" (and first session starts with it)
    note1: assumes certain order of capturing in telaviv format (since no names are given)
    note2: in tel-aviv format, two scalars in the beginning of the file are assumed to be skull sizes of subject.
    :param template_path: the path to template file
    :param input_file_format: force reading file using a specific format (if this is None tries to infer format by content)
    :return: positions is a list of np array per session, names is a list of lists of names per session.
             note: if two sensors exists, they are stacked in a nx2x3 array, else nx3 for positions.
    """
    file_handle = open(str(template_path))
    file_contents = file_handle.read()
    contents_split = file_contents.splitlines()
    non_empty_lines = [line for line in contents_split if line]
    delimiters = [i for i, x in enumerate(non_empty_lines) if x == "*"]
    names = [[]]
    skulls = None
    if not delimiters:
        cond = len(non_empty_lines[0].split()) <= 4
        sessions = [non_empty_lines]
    else:
        cond = len(non_empty_lines[delimiters[0]+1].split()) <= 4
        sessions = [non_empty_lines[delimiters[i]+1:delimiters[i+1]] for i in range(len(delimiters)-1)]
        sessions += [non_empty_lines[delimiters[-1]+1:]]
        # sessions = [non_empty_lines[delimiters[0]+1:delimiters[1]],
        #             non_empty_lines[delimiters[1]+1:delimiters[2]],
        #             non_empty_lines[delimiters[2]+1:]]
        skulls = []
        for x in non_empty_lines[0:delimiters[0]]:
            skull = re.findall(r"[-+]?\d*\.\d+|\d+", x)
            if skull:
                skull = float(skull[0])
                skulls.append(skull)
        if skulls:
            skulls = np.array(skulls)
            skulls = np.mean(skulls)
        else:
            skulls = None
        names = [list() for _ in range(len(sessions))]
    if input_file_format:
        file_format = input_file_format
        if file_format == "princeton":
            if "***" in non_empty_lines[0]:
                non_empty_lines.pop(0)
    else:
        if cond:
            file_format = "princeton"
            if "***" in non_empty_lines[0]:
                non_empty_lines.pop(0)
        else:
            file_format = "telaviv"
    data = []
    if file_format == "telaviv":
        labeled_names = ['lpa', 'nz', 'nosetip', 'lefteye', 'righteye', 'rpa',
                         'f8', 'fp2', 'fpz', 'fp1', 'f7', 'cz', 'o1', 'oz', 'o2']
        for j, session in enumerate(sessions):
            sensor1_data = []
            sensor2_data = []
            for i, (sens1, sens2) in enumerate(pairwise(session)):
                if i < len(labeled_names):
                    name = labeled_names[i]
                else:
                    name = i-len(labeled_names)
                data1 = [x for x in sens1.replace(" ", ",").split(",") if x]
                if data1[1] == "?":
                    continue
                names[j].append(name)
                x, y, z = float(data1[1]), float(data1[2]), float(data1[3])
                sensor1_data.append(np.array([x, y, z]))
                data2 = [x for x in sens2.replace(" ", ",").split(",") if x]
                x, y, z = float(data2[1]), float(data2[2]), float(data2[3])
                sensor2_data.append(np.array([x, y, z]))
            data.append(np.stack((sensor1_data, sensor2_data), axis=1))
    elif file_format == "telaviv2":
        labeled_names = ['lpa', 'nz', 'nosetip', 'lefteye', 'righteye', 'rpa',
                         'f8', 'fp2', 'middle_triangle', 'fp1', 'f7', 'cz', 'o1', 'oz', 'o2']
        for j, session in enumerate(sessions):
            sensor1_data = []
            sensor2_data = []
            for i, sens1 in enumerate(session):
                if i < len(labeled_names):
                    name = labeled_names[i]
                else:
                    name = i-len(labeled_names)
                s = re.sub(r"\s+", ',', sens1)
                data1 = [x for x in s.split(",") if x]
                if data1[0] == "?":
                    continue
                names[j].append(name)
                x, y, z = float(data1[0]), float(data1[1]), float(data1[2])
                sensor1_data.append(np.array([x, y, z]))
                sensor2_data.append(np.array([0, 0, 0]))
            data.append(np.stack((sensor1_data, sensor2_data), axis=1))
    else:  # princeton
        for line in non_empty_lines:
            try:
                name, x, y, z = line.split()
            except ValueError:
                name, x, y, z = line.split(",")
            x = float(x)
            y = float(y)
            z = float(z)
            data.append(np.array([x, y, z]))
            try:
                name = int(name)
            except ValueError as verr:
                name = name.lower()
            names[0].append(name)
        data = [np.array(data)]
    return names, data, file_format, skulls


def delete_content_of_folder(folder_path, subfolders_also=True):
    for x in folder_path.glob("*"):
        if x.is_file():
            x.unlink()
        elif x.is_dir():
            if subfolders_also:
                x.rmdir()


def is_process_active(process_name):
    """
    Checks if a process with the given name is running on Linux.
    
    :param process_name: Name of the process to check.
    :return: True if the process is running, False otherwise.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-x", process_name],  # Linux command
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def save_results(data, output_file):
    """
    saves data into output file
    :param data:
    :param output_file:
    :param v:
    :return:
    """
    if data is None:
        return
    if not output_file:
        output_file = "output.txt"
    logging.info("Saving result to output file: " + str(output_file))
    optode_number = len(data[0])
    with open(str(output_file), 'a') as f:
        for i in range(optode_number):
            my_line = "{} {:.3f} {:.3f} {:.3f}\n".format(data[0][i], data[1][i, 0], data[1][i, 1], data[1][i, 2])
            f.write(my_line)


def extract_session_data(file, use_scale=None, scale_by_z=False):
    timesteps_per_sample = 0
    session = open(file, 'r')
    number_of_features = 0
    # cap_scale_max = 1.2
    # cap_scale_min = 0.8
    x_session = []
    sticker_count = 0
    for i, line in enumerate(session):
        timesteps_per_sample += 1
        my_dict = json.loads(line)
        sticker_3d_locs = my_dict["stickers_locs"]
        valid_stickers = my_dict["valid_stickers"]
        cap_rotation = my_dict["cap_rot"]
        rescalex = 1. / 960
        rescaley = 1. / 540
        sticker_2d_locs = [[member[0]['x'] * rescalex, member[0]['y'] * rescaley] for (i, member) in enumerate(zip(sticker_3d_locs, valid_stickers))]
        if i == 0:
            number_of_features = len(sticker_2d_locs) * 2
        for j in range(len(valid_stickers)):
            if not valid_stickers[j]:
                sticker_2d_locs[j] = [0, 0]
            else:
                sticker_count += 1
        # mask facial landmarks if one is missing, also mask if frame is above 6 (no facial landmarks expected there)
        if sticker_2d_locs[0] == [0, 0] or sticker_2d_locs[1] == [0, 0] or sticker_2d_locs[2] == [0, 0] or i >= 5:
            sticker_2d_locs[0] = [0, 0]
            sticker_2d_locs[1] = [0, 0]
            sticker_2d_locs[2] = [0, 0]
        if cap_rotation['x'] > 180:
            cap_rotation['x'] -= 360
        if cap_rotation['y'] > 180:
            cap_rotation['y'] -= 360
        if cap_rotation['z'] > 180:
            cap_rotation['z'] -= 360
        if use_scale:
            # cap_scalex = (cap_scalex - cap_scale_min) / (cap_scale_max - cap_scale_min)
            # cap_scalez = (cap_scalez - cap_scale_min) / (cap_scale_max - cap_scale_min)
            cap_scale = []
            if "x" in use_scale:
                if scale_by_z:
                    xterm = my_dict["scalex"] / my_dict["scalez"]
                else:
                    xterm = my_dict["scalex"]
                cap_scale.append(xterm)
            if "y" in use_scale:
                if scale_by_z:
                    yterm = my_dict["scaley"] / my_dict["scalez"]
                else:
                    yterm = my_dict["scaley"]
                cap_scale.append(yterm)
            if "z" in use_scale:
                if scale_by_z:
                    zterm = 1.0
                else:
                    zterm = my_dict["scalez"]
                cap_scale.append(zterm)
            cap_rots = [cap_rotation['x'], cap_rotation['y'], cap_rotation['z']]
            for term in cap_scale:
                cap_rots.append(term)
        else:
            cap_rots = [cap_rotation['x'], cap_rotation['y'], cap_rotation['z']]
        x_session.append(sticker_2d_locs)
    if sticker_count >= 20:  # remove datapoints with less than 20 sticker occurrences
        x_session = np.reshape(x_session, (timesteps_per_sample, number_of_features))
        y_session = np.array(cap_rots)
    else:
        x_session = None
        y_session = None
    return x_session, y_session


def load_raw_json_db(db_path, use_scale=None, scale_by_z=False):
    """
    loads data from folder containing json files in the format defined by synthetic data renderer
    :param db_path: path to folder
    :param use_scale: whether or not the db contains scale or just rotation angels as labels
    :return: X - a nx10x14 numpy array of floats containing location of 7 stickers in the synthetic images (10 frames per sample)
             Y - the labels of the 10 frames sequences (1x3 euler angels)
    """
    X = []
    Y = []
    for file in sorted(db_path.glob("*.json")):
        x, y = extract_session_data(file, use_scale=use_scale, scale_by_z=scale_by_z)
        if x is not None:
            X.append(x)
            Y.append(y)
    timesteps_per_sample = X[0].shape[0]
    number_of_features = X[0].shape[1]
    X = np.reshape(X, (len(X), timesteps_per_sample, number_of_features))
    Y = np.array(Y)
    return X, Y


def load_db(db_path, format="pickle", filter=None):
    """
    loads a db according to known format
    note: formats supported are either pickle or folder of json serialized files.
    :param db_path: the path to the db file / folder of db files
    :param format: the format (pickle / json)
    :param filter: a list of files used to select specific files
    :return:
    """
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
            number_of_samples = 30000
            skip_files = 0
            count = 0
            for i, file in enumerate(db_path.glob("*.json")):
                if filter:
                    if file.name in filter:
                        x, y = extract_session_data(file, use_scale=False)
                        x = np.expand_dims(x, axis=0)
                        x[:, :, 0::2] *= 960
                        x[:, :, 1::2] *= 540
                        db[file.name] = [{"data": x, "label": y}]
                else:
                    if i < skip_files:
                        continue
                    x, y = extract_session_data(file, use_scale=False)
                    if x is not None:
                        x = np.expand_dims(x, axis=0)
                        x[:, :, 0::2] *= 960
                        x[:, :, 1::2] *= 540
                        db[file.name] = [{"data": x, "label": y}]
                        count += 1
                    if count >= number_of_samples:
                        break
    return db


def serialize_data(file_path, x_train, x_val, y_train, y_val, x_test=None, y_test=None):
    f = open(file_path, 'wb')
    if x_test is None or y_test is None:
        pickle.dump([x_train, x_val, y_train, y_val], f)
    else:
        pickle.dump([x_train, x_val, y_train, y_val, x_test, y_test], f)
    f.close()


def deserialize_data(file_path, with_test_set=True):
    f = open(file_path, 'rb')
    if with_test_set:
        x_train, x_val, y_train, y_val, x_test, y_test = pickle.load(f)
        f.close()
        return x_train, x_val, y_train, y_val, x_test, y_test
    else:
        x_train, x_val, y_train, y_val = pickle.load(f)
        f.close()
        return x_train, x_val, y_train, y_val


def load_from_pickle(pickle_file_path):
    f = open(pickle_file_path, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def dump_to_pickle(pickle_file_path, data):
    f = open(pickle_file_path, 'wb')
    pickle.dump(data, f)
    f.close()


def dump_full_db(db, path=None):
    if path:
        pickle_path = path
    else:
        pickle_path = Path.joinpath(Path("cache"), "full_db.pickle")
    dump_to_pickle(pickle_path, db)


def load_full_db(db_path=None):
    if db_path is None:
        pickle_path = Path.joinpath(Path("cache"), "full_db.pickle")
    else:
        pickle_path = db_path
    if pickle_path.is_file():
        db = load_from_pickle(pickle_path)
    else:
        db = {}
    return db


def move(src, dst):
    shutil.move(src, dst)


def read_digitizer_multi_noptodes_experiment_file(exp_file_loc):
    path = Path(exp_file_loc)
    file_handle = open(str(path))
    file_contents = file_handle.read()
    contents_split = file_contents.splitlines()
    non_empty_lines = [line for line in contents_split if line]
    delimiters = [i for i, x in enumerate(non_empty_lines) if x == "*"]
    data = []
    for i in range(len(delimiters)-1):
        session = non_empty_lines[delimiters[i]+1:delimiters[i+1]]
        sensor1_data = []
        sensor2_data = []
        for sens1, sens2 in pairwise(session):
            data1 = sens1.split()
            x, y, z = float(data1[1]), float(data1[2]), float(data1[3])
            sensor1_data.append(np.array([x, y, z]))
            data2 = sens2.split()
            x, y, z = float(data2[1]), float(data2[2]), float(data2[3])
            sensor2_data.append(np.array([x, y, z]))
        sensor_data = np.abs(np.array(sensor1_data) - np.array(sensor2_data))
        data.append(sensor_data)
    return data
