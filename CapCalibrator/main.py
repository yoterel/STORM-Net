import argparse
import sys
from pathlib import Path
import video
import predict
import geometry
from file_io import save_results


def parse_arguments():
    parser = argparse.ArgumentParser(description='Calibrates fNIRS sensors location based on video.')
    parser.add_argument("video", help="The path to the video file to calibrate sensors with.")
    parser.add_argument("model", help="The base model file path given in space delimited csv format of size nx3.")
    parser.add_argument("-gt", "--ground_truth", help="The ground truth file path to compare results to given in space delimited csv format of size nx3.")
    parser.add_argument("-m", "--mode", type=str, choices=["manual", "semi-auto", "auto", "special"],
                        default="semi-auto",
                        help="Controls whether to automatically or manually annotate the stickers in the video.")
    parser.add_argument("-o", "--output_file", help="The output csv file with calibrated results (given in MNI coordinates)")
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=2, help="Selects verbosity level")
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    cmd_line = 'E:/University/masters/CapTracking/videos/telaviv/yaara/seesion1/1.mp4 E:/Src/CapCalibrator/example_models/example_model2.txt -gt E:/University/masters/CapTracking/videos_data/yaara -m special -v 1'.split()
    # cmd_line = '/disk1/yotam/capnet/openPos/openPos55/GX011592.MP4 /disk1/yotam/capnet/openPos/openPos/openPos50 -m special -gt /disk1/yotam/capnet/openPos/openPos55'.split()
    # cmd_line = '/disk1/yotam/capnet/openPos/real_babies/1778b/GX011447.MP4 /disk1/yotam/capnet/openPos/openPos/openPos50 -m manual -v 1'.split()
    args = parser.parse_args(cmd_line)
    args.video = Path(args.video)
    # if Path.is_dir(args.video):
    #     args.video = args.video.glob("*.MP4").__next__()
    args.model = Path(args.model)
    if Path.is_dir(args.model):
        args.model = args.model.glob("*.txt").__next__()
    if args.ground_truth:
        args.ground_truth = Path(args.ground_truth)
        # if Path.is_dir(args.ground_truth):
        #     args.ground_truth = args.ground_truth.glob("*.txt").__next__()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    sticker_locations = video.process_video(args)  # nx10x14 floats
    r_matrix, s_matrix = predict.predict_rigid_transform(sticker_locations, args)
    sensor_locations = geometry.apply_rigid_transform(r_matrix,
                                                      s_matrix,
                                                      args.model,
                                                      args.ground_truth,
                                                      plot=True,
                                                      v=args.verbosity)
    projected_data = geometry.project_sensors_to_MNI(sensor_locations, args.verbosity)
    save_results(projected_data, args.output_file, args.verbosity)
    if args.verbosity:
        print("Done!")

