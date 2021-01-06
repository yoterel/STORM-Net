import argparse
from pathlib import Path
import subprocess
import sys
from file_io import read_template_file
from geometry import to_standard_coordinate_system
from geometry import fix_yaw
import logging


def parse_arguments():
    parser = argparse.ArgumentParser(description='Renders training images for fNIRS alighment')
    parser.add_argument("template", help="Path to raw template file. For exact format see documentation.")
    parser.add_argument("output", help="A path to a folder where rendered images / data will be created")
    parser.add_argument("--iterations", type=int, default=20, help="Number of rendering iterations (10 images from each)")
    parser.add_argument("--exe", default="renderer.exe", help="The path to the renderer executable.")
    parser.add_argument("--log", default="log.txt", help="The path to the output log file from renderer.")
    parser.add_argument("--no_transform", default=False, action='store_true', help="If specified, the data from the input template model will *NOT* be transformed to standard coordinate system before rendering. This is not recommended.")
    parser.add_argument("--save_images", default=False, action='store_true', help="Renderer will output images (in addition to formatted data)")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
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


def create_temporary_template(names, data, template_file):
    template_file.parent.mkdir(parents=True, exist_ok=True)
    f = open(str(template_file), "w+")
    for i, name in enumerate(names):
        line = "{} {:.3f} {:.3f} {:.3f}\n".format(name, data[i, 0], data[i, 1], data[i, 2])
        f.write(line)
    f.close()


def launch_renderer(exe_path, log_path, iterations, template, output, images):
    cmd = str(exe_path) + \
          " -logFile {}".format(str(log_path.resolve())) +\
          " -iterations {}".format(iterations) +\
          " -input_file {}".format(str(template.resolve())) +\
          " -output_folder {}".format(str(output.resolve())) +\
          " -batchmode"
    if images:
        cmd += " -save_image True"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    logging.info("Renderer launched as daemon.")


def render(template_names, template_data, output_folder, exe_path, log_path, iterations, images):
    data = to_standard_coordinate_system(template_names, template_data)
    data = fix_yaw(template_names, data)
    template_file_path = Path("cache", "template_transformed.txt")
    create_temporary_template(template_names, data, template_file_path)
    launch_renderer(exe_path, log_path, iterations, template_file_path, output_folder, images)
    
    
if __name__ == "__main__":
    args = parse_arguments()
    names, data, file_format, _ = read_template_file(args.template)
    data = data[0]  # select first (and only) session
    if file_format == "telaviv":
        data = data[:, 0, :]  # select first sensor
    names = names[0]  # select first (and only) session
    if not args.no_transform:
        render(names, data, args.output, args.exe, args.log, args.iterations, args.save_images)
    else:
        launch_renderer(args.exe, args.log, args.iterations, args.template, args.output, args.images)
    logging.info("See", args.log.resolve(), "for detailed renderer log.")
