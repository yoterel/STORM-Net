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
    # cmd_line = './../example_models/example_model2.txt ./build/output --exe ./build/UnityCap.exe --log ./build/log.txt --transform_input --iterations 30000'.split()
    args = parser.parse_args()
    args.exe = Path(args.exe)
    args.log = Path(args.log)
    args.template = Path(args.template)
    args.output = Path(args.output)
    if args.output.is_dir():
        print("Warning! output folder already exists (data will be overridden) ok?")
        val = input("y / n\n")
        if val == "n":
            print("Aborting renderer launch.")
            exit()
    args.output.mkdir(parents=True, exist_ok=True)
    return args


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
        print("Warning ! Z axis is pointing downwards, fixing.")
        if (xdir > 0 and ydir > 0) or (xdir < 0 and ydir < 0):
            return [1, 1, -1]
        else:
            if xdir > 0:
                return [1, -1, -1]
            else:
                return [-1, 1, -1]


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
    yaw_vec_3 = (data[Fp1] - data[Fp2]) * np.array([1, 1, 0])
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
    f = open("template_transformed.txt", "w+")
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
    names, data = read_file(args.template)
    if args.transform:
        flip_axis = check_right_handed_system(names, data)
        data[:, 0] *= flip_axis[0]
        data[:, 1] *= flip_axis[1]
        data[:, 2] *= flip_axis[2]
        data = fix_yaw(names, data)
        save_intermediate(names, data)
        args.template = Path("template_transformed.txt")
    launch_renderer(args)
    print("See", args.log.resolve(), "for detailed renderer log.")
