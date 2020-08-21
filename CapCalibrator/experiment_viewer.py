import pptk
import argparse
from pathlib import Path
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='Renders training images for fNIRS alighment')
    parser.add_argument("template", help="Path to raw template file. For exact format see documentation.")
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    cmd_line = "E:/Src/CapCalibrator/CapCalibrator/experiment_model.txt".split()
    args = parser.parse_args(cmd_line)
    args.template = Path(args.template)
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


if __name__ == "__main__":
    args = parse_arguments()
    names, data = read_file(args.template)
    color = data[:, 2] / np.max(data[:, 2])
    color = np.expand_dims(color, axis=0)
    color = np.repeat(color, 3, axis=0)
    color = color.T
    color_rand = pptk.rand(len(data), 3)
    v = pptk.viewer(data)
    data_mean = np.mean(data, axis=0)
    v.attributes(color, color_rand)
    v.set(point_size=0.1)
    v.set(lookat=data_mean)
    v.set(r=20)
    for i in range(1, len(data)+1):
        v.set(selected=i-1)
        print("Point:", i)
        v.wait()
    v.close()

