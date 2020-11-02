import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # select GPU indexing to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # select GPU to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress warnings & info from tf
import argparse
from pathlib import Path
import video
import predict
import geometry
from file_io import save_results
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description='Calibrates fNIRS sensors location based on video.')
    parser.add_argument("-m", "--mode", type=str, choices=["semi-auto", "auto", "experimental"],
                        default="semi-auto",
                        help="Controls operation mode, in semi-auto/auto mode GUI is launched/not launched.")
    parser.add_argument("-vid", "--video", help="The path to the video file to calibrate sensors with. Required if mode is auto.")
    parser.add_argument("-t", "--template", help="The template file path (given in space delimited csv format of size nx3). Required if mode is auto")
    parser.add_argument("-stormnet", "--storm_net", default="telaviv_model_b16.h5", help="A path to a trained storm net keras model")
    parser.add_argument("-unet", "--u_net", default="unet_tel_aviv.h5",
                        help="A path to a trained segmentation network model")
    parser.add_argument("-s", "--session_file",
                        help="A file containing processed results for previous videos.")
    parser.add_argument("-out", "--output_file", help="The output csv file with calibrated results (given in MNI coordinates)")
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=2, help="Selects verbosity level")
    parser.add_argument("-gt", "--ground_truth", help="Use this in experimental mode only")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    if args.mode == "semi-auto":
        args.video = None
        args.template = None
        args.output_file = None
        args.session_file = None
    else:
        args.video = Path(args.video)
        args.template = Path(args.template)
        args.session_file = Path(args.session_file)
        if Path.is_dir(args.template):
            args.template = args.template.glob("*.txt").__next__()
        if args.ground_truth:
            args.ground_truth = Path(args.ground_truth)
    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.mode == "semi-auto":
        video.process_video(args)
    else:
        sticker_locations, video_names = video.process_video(args)
        r_matrix, s_matrix = predict.predict_rigid_transform(sticker_locations, None, None, args)
        sensor_locations = geometry.apply_rigid_transform(r_matrix, s_matrix, None, None, video_names, args)
        projected_data = geometry.project_sensors_to_MNI(sensor_locations, args.verbosity)
        save_results(projected_data, args.output_file, args.verbosity)
