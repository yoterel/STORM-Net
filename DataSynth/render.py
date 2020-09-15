import argparse
from pathlib import Path
import numpy as np
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser(description='Renders training images for fNIRS alighment')
    parser.add_argument("template", help="Path to raw template file. For exact format see documentation.")
    parser.add_argument("output", help="The path where rendered images / data will be created")
    parser.add_argument("--iterations", type=int, default=20, help="Number of rendering iterations (10 images from each)")
    parser.add_argument("--exe", default="renderer.exe", help="The path to the renderer executable.")
    parser.add_argument("--log", default="log.txt", help="The path to the output log file from renderer.")
    parser.add_argument("--transform", "--transform_input", default=False, action='store_true', help="Input template file will be transformed to comply with renderer. Intermediate result will be saved to 'template_transformed.txt'.")
    parser.add_argument("--images", "--save_images", default=False, action='store_true', help="Renderer will output images (in addition to formatted data)")
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    cmd_line = './../example_models/new_magnetic_source_model.txt ./build/output --exe ./build/UnityCap.exe --log ./build/log.txt --transform_input --iterations 20 --images'.split()
    args = parser.parse_args()
    args.exe = Path(args.exe)
    args.log = Path(args.log)
    args.template = Path(args.template)
    args.output = Path(args.output)
    if args.output.is_dir():
        print("Warning! output folder already exists (data will be overridden) ok?")
        val = input("y / n\n")
        if val != "y":
            print("Aborting renderer launch.")
            exit()
    args.output.mkdir(parents=True, exist_ok=True)
    return args


def pairwise(iterable):
    """
    turns an iterable into a pairwise iterable
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    :param iterable:
    :return:
    """
    a = iter(iterable)
    return zip(a, a)


def read_template_file(template_path):
    """
    reads a template file in telaviv format ("sensor x y z rx ry rz") or in princeton format ("name x y z")
    note: assumes certain order of capturing in telaviv format (since no names are given)
    :param template_path: the path to template file
    :return: names & position of points in data (if two sensors exists, they are stacked in a nx2x3 array, else nx3)
    """
    file_handle = open(str(template_path))
    file_contents = file_handle.read()
    contents_split = file_contents.splitlines()
    non_empty_lines = [line for line in contents_split if line]
    if len(non_empty_lines[0].split()) <= 4:
        file_format = "princeton"
        if "***" in non_empty_lines[0]:
            non_empty_lines.pop(0)
    else:
        file_format = "telaviv"
    names = []
    data = []
    if file_format == "telaviv":
        sensor1_data = []
        sensor2_data = []
        labeled_names = ['rightear', 'nosebridge', 'nosetip', 'righteye', 'lefteye', 'leftear', 'cz', 'fp1', 'fp2', 'fpz']
        # labeled_names = [item for item in labeled_names for i in range(2)]
        for i, (sens1, sens2) in enumerate(pairwise(non_empty_lines)):
            if i < len(labeled_names):
                name = labeled_names[i]
            else:
                name = i-len(labeled_names)
            names.append(name)
            data1 = sens1.split()
            x, y, z = float(data1[1]), float(data1[2]), float(data1[3])
            sensor1_data.append(np.array([x, y, z]))
            data2 = sens2.split()
            x, y, z = float(data2[1]), float(data2[2]), float(data2[3])
            sensor2_data.append(np.array([x, y, z]))
        data = np.stack((sensor1_data, sensor2_data), axis=1)
    else:  # princeton
        for line in contents_split:
            name, x, y, z = line.split()
            x = float(x)
            y = float(y)
            z = float(z)
            data.append(np.array([x, y, z]))
            try:
                name = int(name)
            except ValueError as verr:
                name = name.lower()
            names.append(name)
        if 0 not in names:
            end = names[-1]
            names[names.index(1):] = [x for x in range(end)]
        data = np.array(data)
    return names, data, file_format


def read_file(template_path):
    file_handle = open(str(template_path))
    file_contents = file_handle.read()
    contents_split = file_contents.splitlines()
    data = []
    names = []
    for line in contents_split:
        name, x, y, z = line.split()
        x = float(x)
        y = float(y)
        z = float(z)
        data.append(np.array([x, y, z]))
        names.append(name.lower())
    data = np.array(data)
    return names, data


def check_right_handed_system(names, data):
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
        if (xdir > 0 and ydir > 0) or (xdir < 0 and ydir < 0):
            return [1, 1, -1]
        else:
            if xdir > 0:
                return [1, -1, -1]
            else:
                return [-1, 1, -1]


def check_standard_coordiante_system(names, data):
    """
    swaps axis of data according to standard cordinate system (x is ear-to-ear, y is back-front, z is bottom-top
    note: doesn't fix handedness or sign of axis.
    :param names:
    :param data:
    :return:
    """
    leftEye = names.index('lefteye')
    rightEye = names.index('righteye')
    fpz = names.index('fpz')
    fp1 = names.index('fp1')
    fp2 = names.index('fp2')
    x_axis = np.argmax(np.abs(data[rightEye] - data[leftEye]))
    data[:, [0, x_axis]] = data[:, [x_axis, 0]]
    z_axis = np.argmax(np.abs(data[fpz] - ((data[fp1]+data[fp2]) / 2)))
    if z_axis != 0:
        data[:, [2, z_axis]] = data[:, [z_axis, 2]]
    return data

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
    leftEar = names.index('leftear')
    rightEar = names.index('rightear')
    Fp2 = names.index('fp2')
    Fp1 = names.index('fp1')
    yaw_vec_1 = (data[rightEye] - data[leftEye]) * np.array([1, 1, 0])
    yaw_vec_2 = (data[rightEar] - data[leftEar]) * np.array([1, 1, 0])
    yaw_vec_3 = (data[Fp2] - data[Fp1]) * np.array([1, 1, 0])
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


def save_intermediate(names, data):
    temp_folder_path = Path("temp")
    temp_file_path = Path.joinpath(temp_folder_path, "template_transformed.txt")
    temp_folder_path.mkdir(parents=True, exist_ok=True)
    f = open(str(temp_file_path), "w+")
    for i, name in enumerate(names):
        line = "{} {:.3f} {:.3f} {:.3f}\n".format(name, data[i, 0], data[i, 1], data[i, 2])
        f.write(line)
    f.close()


def launch_renderer(args):
    cmd = str(args.exe) + \
          " -logFile {}".format(str(args.log.resolve())) +\
          " -iterations {}".format(args.iterations) +\
          " -input_file {}".format(str(args.template.resolve())) +\
          " -output_folder {}".format(str(args.output.resolve())) +\
          " -batchmode"
    if args.images:
        cmd += " -save_image True"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    print("Renderer launched as daemon.")


if __name__ == "__main__":
    args = parse_arguments()
    names, data, format = read_template_file(args.template)
    data = data[:, 0, :]
    if args.transform:
        data = check_standard_coordiante_system(names, data)
        flip_axis = check_right_handed_system(names, data)
        data[:, 0] *= flip_axis[0]
        data[:, 1] *= flip_axis[1]
        data[:, 2] *= flip_axis[2]
        data = fix_yaw(names, data)
        save_intermediate(names, data)
        args.template = Path("temp", "template_transformed.txt")
    launch_renderer(args)
    print("See", args.log.resolve(), "for detailed renderer log.")
