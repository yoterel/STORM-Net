import utils
import argparse
from pathlib import Path
import video
import predict
import geometry
import logging
from file_io import save_results
import experimental
import sys
__version__ = "0.0.1"


def parse_arguments():
    parser = argparse.ArgumentParser(description='Coregisters fNIRS sensors location based on video.')
    parser.add_argument("--mode", type=str, choices=["gui", "auto", "experimental"],
                        default="gui",
                        help="Controls operation mode of application.")
    parser.add_argument("--video", help="The path to the video file to coregister sensors with. Required if mode is auto.")
    parser.add_argument("--template", help="The template file path (given in space delimited csv format of size nx3). Required if mode is auto")
    parser.add_argument("--mni", action="store_true",
                        help="If specified, output will be projected to (adult) MNI coordinates")
    parser.add_argument("--storm_net", default="models/torch_heatmap_manuscript.h5", help="A path to a trained storm net model")
    parser.add_argument("--unet", help="A path to a trained segmentation network model")
    parser.add_argument("--session_file",
                        help="A file containing processed results for previous videos.")
    parser.add_argument("--output_file", help="The output csv file with coregistered results")
    parser.add_argument("--verbosity", type=str, choices=["debug", "info", "warning"], default="info", help="Selects verbosity level")
    parser.add_argument("--log", help="If specified, log will be output to this file")
    parser.add_argument("--ground_truth", help="Use this in experimental mode only")
    parser.add_argument("--gpu_id", type=int, default=-1, help="Which GPU to use (or -1 for cpu)")
    parser.add_argument("--headless", action="store_true",
                        help="Force no gui")
    parser.add_argument("--version", action="store_true", help="Prints version")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    # configure computing environment
    if args.version:
        print(__version__, end="")
        sys.exit(0)
    if args.unet is None:
        args.unet = Path(Path(__file__).parent, "models", "unet_tel_aviv.h5")
    args.device = utils.configure_compute_environment(args.gpu_id)
    if args.mode == "gui" and not args.headless:
        args.video = None
        args.template = None
        args.output_file = None
        args.session_file = None
        args.ground_truth = None
        args.storm_net = None
    else:
        if args.video:
            args.video = Path(args.video)
        else:
            logging.error("Missing video file parameter")
            exit(1)
        if args.template:
            args.template = Path(args.template)
            if Path.is_dir(args.template):
                args.template = args.template.glob("*.txt").__next__()
        else:
            logging.error("Missing template file parameter")
            exit(1)
        if args.session_file:
            args.session_file = Path(args.session_file)
        if args.ground_truth:
            args.ground_truth = Path(args.ground_truth)
        if args.output_file:
            args.output_file = Path(args.output_file)
        if not Path(args.storm_net).is_file():
            logging.error("Storm-Net model file not found (using: {})".format(args.storm_net))
            exit(1)
        if not Path(args.unet).is_file():
            logging.error("Semantic segmentation model file not found (using: {})".format(args.unet))
            exit(1)
    return args


if __name__ == "__main__":
    # parse command line
    args = parse_arguments()
    # set up logging
    if args.log:
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())
    # run GUI / automatic annotation
    sticker_locations, video_names = video.process_video(args)
    if args.mode == "auto":
        r_matrix, s_matrix = predict.predict_rigid_transform(sticker_locations, None, args)
        sensor_locations = geometry.apply_rigid_transform(r_matrix, s_matrix, None, None, video_names, args)
        if args.mni:
            projected_data = geometry.project_sensors_to_MNI(sensor_locations)
        else:
            projected_data = sensor_locations
        save_results(projected_data[0], args.output_file)
    elif args.mode == "experimental":
        experimental.reproduce_experiments(video_names, sticker_locations, args)
