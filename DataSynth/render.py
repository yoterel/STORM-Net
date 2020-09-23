import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, '../CapCalibrator')
from file_io import read_template_file
from geometry import to_standard_coordinate_system
from geometry import fix_yaw


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
    cmd_line = './../raw_data/new_magnetic_source_model.txt ./build/output --exe ./build/UnityCap.exe --log ./build/log.txt --transform_input --iterations 50 --images'.split()
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
    data = data[0]  # select first (and only) session
    data = data[:, 0, :]  # select first sensor
    names = names[0]  # select first (and only) session
    if args.transform:
        data = to_standard_coordinate_system(names, data)
        data = fix_yaw(names, data)
        save_intermediate(names, data)
        args.template = Path("temp", "template_transformed.txt")
    launch_renderer(args)
    print("See", args.log.resolve(), "for detailed renderer log.")
