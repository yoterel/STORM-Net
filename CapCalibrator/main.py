import argparse
import sys
from pathlib import Path
import video
import predict
import numpy as np
import geometry


def parse_arguments():
    parser = argparse.ArgumentParser(description='Calibrates fNIRS sensors location based on video.')
    parser.add_argument("video", help="The path to the video file to calibrate sensors with.")
    parser.add_argument("model", help="The base model file path given in space delimited csv format of size nx3.")
    parser.add_argument("-gt", "--ground_truth", help="The ground truth file path to compare results to given in space delimited csv format of size nx3.")
    parser.add_argument("-a", "--automation_level", type=str, choices=["manual", "semi-auto", "auto"],
                        default="semi-auto",
                        help="Controls whether to automatically or manually annotate the stickers in the video.")
    parser.add_argument("-o", "--output_file", help="The output csv file with calibrated results (given in MNI coordinates)")
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=2, help="Selects verbosity level")
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    # cmd_line = 'E:/University/masters/CapTracking/videos/openpos20/GX011434.MP4 E:/University/masters/CapTracking/videos/openpos18/openPos18.txt -a manual -gt E:/University/masters/CapTracking/videos/openpos20/openPos20.txt'.split()
    # cmd_line = 'E:/University/masters/CapTracking/videos/openpos19/GX011433.MP4 E:/University/masters/CapTracking/videos/openpos18/openPos18.txt -a manual -gt E:/University/masters/CapTracking/videos/openpos19/openPos19.txt'.split()
    # cmd_line = 'E:/University/masters/CapTracking/videos/openpos26/GoPro.MP4 E:/University/masters/CapTracking/videos/openpos25/openPos25.txt -a manual -gt E:/University/masters/CapTracking/videos/openpos26/openPos26.txt'.split()
    # cmd_line = 'E:/University/masters/CapTracking/videos/openpos28/GX011444_Trim.mp4 E:/University/masters/CapTracking/videos/openpos27/openPos27.txt -a manual -gt E:/University/masters/CapTracking/videos/openpos28/openPos28.txt'.split()
    # cmd_line = 'E:/University/masters/CapTracking/videos/openpos54 E:/University/masters/CapTracking/videos/openpos50 -a manual -gt E:/University/masters/CapTracking/videos/openpos54'.split()
    cmd_line = 'E:/University/masters/CapTracking/videos/test/GX011559.MP4 E:/University/masters/CapTracking/videos/test_model -a manual -gt E:/University/masters/CapTracking/videos/test'.split()
    args = parser.parse_args(cmd_line)
    args.video = Path(args.video)
    # if Path.is_dir(args.video):
    #     args.video = args.video.glob("*.MP4").__next__()
    args.model = Path(args.model)
    if Path.is_dir(args.model):
        args.model = args.model.glob("*.txt").__next__()
    args.ground_truth = Path(args.ground_truth)
    if Path.is_dir(args.ground_truth):
        args.ground_truth = args.ground_truth.glob("*.txt").__next__()
    return args


def save_results(projected_data, output_file, v):
    if v:
        print("Saving result to output file.")
    if not output_file:
        output_file = "output.txt"
    # np.savetxt(output_file, projected_data, delimiter=" ")


if __name__ == "__main__":
    args = parse_arguments()
    sticker_locations = video.process_video(args.video, args.automation_level, args.verbosity)  # nx10x14 floats
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

