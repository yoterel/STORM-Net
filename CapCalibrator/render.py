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
    parser.add_argument("--iterations", type=int, default=10, help="Number of rendering iterations (10 images from each)")
    parser.add_argument("--exe", default="renderer.exe", help="The path to the renderer executable.")
    parser.add_argument("--log", default="log.txt", help="The path to the output log file from renderer.")
    parser.add_argument("--no_transform", default=False, action='store_true', help="If specified, the data from the input template model will *NOT* be transformed to standard coordinate system before rendering. This is not recommended.")
    parser.add_argument("--save_images", default=False, action='store_true', help="Renderer will output images (in addition to formatted data)")
    parser.add_argument("--scale_faces", type=str, choices=["x", "y", "z", "xy", "xz", "yz", "xyz"], help="Renderer will also apply different scales to the virtual head & mask")
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


def launch_renderer(exe_path, log_path, iterations, template, output, images, scale_faces):
    # breakpoint()
    cmd = [
        str(exe_path), 
        "-logFile", str(log_path.resolve()),
        "-iterations", str(iterations),
        "-input_file", str(template.resolve()),
        "-output_folder", str(output.resolve()),
        "-batchmode",
        "-nographics"
    ]
    if scale_faces:
        cmd.extend(["-scale", str(scale_faces)])
    if images:
        cmd.extend(["-save_image", "True"])
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        logging.info("Renderer launched as daemon.")
        return True, process
    except FileNotFoundError:
        logging.error("Failed to run renderer. Does it really exist in this path? " + str(exe_path))
        return False, None


def render(template_names, template_data, output_folder, exe_path, log_path, iterations, images, scale_faces):
    data = fix_yaw(template_names, template_data)
    data = to_standard_coordinate_system(template_names, data)
    template_file_path = Path("cache", "template_transformed.txt")
    create_temporary_template(template_names, data, template_file_path)
    success, process = launch_renderer(exe_path, log_path, iterations, template_file_path, output_folder, images, scale_faces)
    return success, process


if __name__ == "__main__":
    args = parse_arguments()
    names, data, file_format, _ = read_template_file(args.template)
    data = data[0]  # select first (and only) session
    if file_format == "telaviv":
        data = data[:, 0, :]  # select first sensor
    names = names[0]  # select first (and only) session
    if not args.no_transform:
        render(names, data, args.output, args.exe, args.log, args.iterations, args.save_images, args.scale_faces)
    else:
        launch_renderer(args.exe, args.log, args.iterations, args.template, args.output, args.images, args.scale_faces)
    logging.info("See", args.log.resolve(), "for detailed renderer log.")
