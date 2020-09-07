import numpy as np
import imageio
from PIL import Image, ImageTk
import video_annotator
from pathlib import Path
import geometry

def select_frames(vid_path, steps_per_datapoint=10, starting_frame=0, frame_indices=None):
    # Read an image, a window and bind the function to window
    # cap = cv2.VideoCapture(str(video_path))
    reader = imageio.get_reader(vid_path, 'ffmpeg')
    meta_data = reader.get_meta_data()
    estimated_total_frames = np.array(meta_data["fps"] * meta_data["duration"], dtype=int).tolist()
    frames_to_use = estimated_total_frames
    if starting_frame >= (frames_to_use // steps_per_datapoint):
        starting_frame = 0
    # db = np.zeros((frames_to_use // steps_per_datapoint, steps_per_datapoint, number_of_features))
    # imgs = np.zeros((steps_per_datapoint, 960, 540))
    # my_dict = {"db": db, "img": img}
    frames = []
    indices = []
    for i, im in enumerate(reader):
        if frame_indices is not None:
            if i in frame_indices:
                frames.append(Image.fromarray(im).resize((960, 540)))
                indices.append(i)
        else:
            if i >= frames_to_use:
                break
            else:
                if i % (frames_to_use // steps_per_datapoint) == starting_frame:
                    frames.append(Image.fromarray(im).resize((960, 540)))
                    indices.append(i)
                    if len(frames) >= steps_per_datapoint:
                        break
    return frames, indices


def process_video(args):
    vid_path = args.video
    mode = args.mode
    v = args.verbosity
    new_db = []
    if mode == "special":
        if v:
            print("Doing special stuff.")
        geometry.print_calibration_results(args.model, args.ground_truth)
        # new_db = video_annotator.auto_annotate_videos(vid_path, args.model, mode)
        new_db = video_annotator.annotate_videos(vid_path, mode, v)
        data = np.zeros((len(new_db.keys()), 10, 14))
        for i, key in enumerate(sorted(new_db.keys())):
            data[i] = new_db[key]["data"]
        return data
    else:
        if mode == "auto":
            new_db = video_annotator.auto_annotate_videos(vid_path, args.model, mode)
        else:
            if mode == "semi-auto" or mode == "manual":
                if v:
                    print("Launching GUI to manually fix/annotate frames.")
                new_db = video_annotator.annotate_videos(vid_path, mode, v)
        if Path.is_dir(vid_path):
            vid_names = []
            for file in sorted(vid_path.glob("*.MP4")):
                vid_names.append(file.name)
            if v:
                print("Found following video files:", vid_names)
            data = np.zeros((len(vid_names), 10, 14))
            for i, vid in enumerate(vid_names):
                data[i] = new_db[vid][0]["data"]
            return data
        else:
            return new_db[vid_path.name][0]["data"]
