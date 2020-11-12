import argparse
from pathlib import Path
import video
import predict
import geometry
import logging
from file_io import save_results
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description='Calibrates fNIRS sensors location based on video.')
    parser.add_argument("-m", "--mode", type=str, choices=["gui", "auto", "experimental"],
                        default="gui",
                        help="Controls operation mode of application")
    parser.add_argument("-vid", "--video", help="The path to the video file to calibrate sensors with. Required if mode is auto.")
    parser.add_argument("-t", "--template", help="The template file path (given in space delimited csv format of size nx3). Required if mode is auto")
    parser.add_argument("-stormnet", "--storm_net", default="models/telaviv_model_b16.h5", help="A path to a trained storm net keras model")
    parser.add_argument("-unet", "--u_net", default="models/unet_tel_aviv.h5",
                        help="A path to a trained segmentation network model")
    parser.add_argument("-s", "--session_file",
                        help="A file containing processed results for previous videos.")
    parser.add_argument("-out", "--output_file", help="The output csv file with calibrated results (given in MNI coordinates)")
    parser.add_argument("-v", "--verbosity", type=str, choices=["debug", "info", "warning"], default="info", help="Selects verbosity level")
    parser.add_argument("-log", "--log", help="If specified, log will be output to this file")
    parser.add_argument("-gt", "--ground_truth", help="Use this in experimental mode only")
    parser.add_argument("--gpu_id", type=int, default=-1, help="Which GPU to use (or -1 for cpu)")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    if args.mode == "gui":
        args.video = None
        args.template = None
        args.output_file = None
        args.session_file = None
        args.ground_truth = None
    else:
        if args.video:
            args.video = Path(args.video)
        if args.template:
            args.template = Path(args.template)
        if args.session_file:
            args.session_file = Path(args.session_file)
        if Path.is_dir(args.template):
            args.template = args.template.glob("*.txt").__next__()
        if args.ground_truth:
            args.ground_truth = Path(args.ground_truth)
    return args


def configure_compute_environment(gpu_id):
    import os
    if gpu_id == -1:
        gpu_id = ""
    else:
        gpu_id = str(gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id  # set gpu visibility prior to importing tf and keras
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress warnings & info from tf
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


if __name__ == "__main__":
    # parse command line
    args = parse_arguments()
    # set up logging
    if args.log:
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())
    # configure computing environment
    configure_compute_environment(args.gpu_id)
    # run GUI / automatic annotation
    if args.mode == "gui":
        video.process_video(args)
    else:
        sticker_locations, video_names = video.process_video(args)
        r_matrix, s_matrix = predict.predict_rigid_transform(sticker_locations, None, None, args)
        sensor_locations = geometry.apply_rigid_transform(r_matrix, s_matrix, None, None, video_names, args)
        projected_data = geometry.project_sensors_to_MNI(sensor_locations)
        save_results(projected_data, args.output_file)
