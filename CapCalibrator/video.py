import numpy as np
import imageio
from PIL import Image
import video_annotator
from pathlib import Path
import utils
import cv2
import file_io
import predict


def select_frames(vid_path, steps_per_datapoint=10, starting_frame=0, local_env_size=5, frame_indices=None):
    """
    selects "steps_per_datapoint" number of frames from a video starting with "starting_frame".
    frames spacing is set uniformly unless local_env_size is greater than 1
    user may also supply his own indices skipping this process.
    :param vid_path: the path to the video file
    :param steps_per_datapoint: the number of frames to retrieve
    :param starting_frame: the starting frame, defualts to 0
    :param local_env_size: the local environment of indices to look for non-blurred images using variance of laplacian
            note: if 1 then selected frames are the initial evenly spaced sites.
    :param frame_indices: if specified, these are the frames that are selected
    :return: the frames (list of PIL images) and their respected indices
    """
    reader = imageio.get_reader(vid_path, 'ffmpeg')
    meta_data = reader.get_meta_data()
    estimated_total_frames = int(meta_data["fps"] * meta_data["duration"])
    frames_to_use = estimated_total_frames
    assert(starting_frame < frames_to_use)
    frames_to_use -= starting_frame
    stride = frames_to_use // steps_per_datapoint
    assert (local_env_size < stride)
    assert (local_env_size % 2 == 1)
    if frame_indices is not None:
        indices = [frame_indices]
    else:
        base_sites = [i for i in range(0, frames_to_use, stride)]
        base_sites = base_sites[:steps_per_datapoint]
        base_sites = [x+starting_frame for x in base_sites]
        indices = []
        for x in base_sites:
            begin, end = utils.get_local_range(x, local_env_size, frames_to_use)
            indices.append([x for x in range(begin, end+1)])
    frames = np.empty(np.array(indices).shape, dtype=object)
    for i, im in enumerate(reader):
        if any(i in sublist for sublist in indices):
            loc = [(j, idx.index(i)) for j, idx in enumerate(indices) if i in idx]
            loc = loc[0]
            frames[loc[0], loc[1]] = Image.fromarray(im).resize((960, 540))
    if local_env_size != 1 and frame_indices is None:
        selected_sites = []
        for i in range(len(frames)):
            cur = frames[i, :]
            blur = np.array([measure_blur(xi) for xi in cur])
            selected_sites.append(np.argmax(blur).astype(int))
        indices = [x[s] for x, s in zip(indices, selected_sites)]
        frames = [x[s] for x, s in zip(frames.tolist(), selected_sites)]
    else:
        frames = frames[0].tolist()
        indices = indices[0]
    return frames, indices


def measure_blur(frame):
    """
    returns a score measureing how blurry is the frame (the higher, the less blurry)
    :param frame: the frame to analyze
    :return: a float representing the score
    """
    return cv2.Laplacian(np.array(frame), cv2.CV_64F).var()


def video_to_frames(vid_path, dump_frames=False, starting_frame=0, force_reselect=False, frame_indices=None, v=0):
    """
    given a video path, splits it into frames and possibly dumps them as a serialized file
    :param vid_path: the video file path
    :param dump_frames: whether or not to dump the frames as a pickle file
    :param starting_frame: the frame to start from when selecting the frames
    :param force_reselect: if true, performs split even if dump file exists
    :param frame_indices: if specified, selects this list of frames indices from the video
    :param v: verbosity
    :return:
    """
    if frame_indices is not None:
        starting_frame = frame_indices[0]
    name = vid_path.parent.name + "_" + vid_path.name
    # if starting_frame != 0:
    #     my_string = name + "_{:03d}_frames.pickle".format(starting_frame)
    # else:
    my_string = name + "_frames.pickle"
    pickle_path = Path.joinpath(Path("data"), my_string)
    if pickle_path.is_file() and not force_reselect:
        frames, indices = file_io.load_from_pickle(pickle_path)
    else:
        if v:
            print("Selecting frames for video:", vid_path)
        frames, indices = select_frames(vid_path,
                                        steps_per_datapoint=10,
                                        starting_frame=starting_frame,
                                        frame_indices=frame_indices)
        if dump_frames:
            file_io.dump_to_pickle(pickle_path, [frames, indices])
    return frames, indices


def process_video(args):
    """
    given video(s) path and mode of operation, process the video and output sticker locations
    :param args:
    :return: sticker locations in a nx14x3 numpy array
    """
    vid_path = args.video
    mode = args.mode
    v = args.verbosity
    new_db = []
    if mode == "experimental":
        if v:
            print("Performing experimental stuff.")
        new_db = video_annotator.annotate_videos(vid_path, mode, args)
    else:
        if mode == "auto":
            new_db = auto_annotate_videos(vid_path, True, True, args)
        else:
            if mode == "semi-auto" or mode == "manual":
                if v:
                    print("Launching GUI to manually fix/annotate frames.")
                new_db = video_annotator.annotate_videos(vid_path, mode, args)
    if Path.is_dir(vid_path):
        vid_names = []
        for file in sorted(vid_path.glob("**/*.MP4")):
            name = file.parent.name + "_" + file.name
            vid_names.append(name)
        if v:
            print("Found following video files:", vid_names)
        data = np.zeros((len(vid_names), 10, 14))
        for i, vid in enumerate(vid_names):
            try:
                data[i] = new_db[vid][0]["data"]
            except KeyError:
                print("Error! did you forget to annotate {} ?".format(vid))
                exit(1)
        return data, vid_names
    else:
        return new_db[vid_path.parent.name + "_" + vid_path.name][0]["data"], [vid_path]


def auto_annotate_videos(vid_path, dump_to_db, force_annotate, args):
    """
    given a video file or folder of video files, automatically annotates the video
    :param vid_path:
    :param model_digi_file:
    :param mode:
    :return:
    """
    db_path = Path("data", "full_db.pickle")
    model_dir = Path("models")
    model_name = args.u_net
    model_full_name = Path.joinpath(model_dir, model_name)
    my_model = file_io.load_semantic_seg_model(str(model_full_name))
    my_db = file_io.load_full_db(db_path)
    paths = []
    if Path.is_file(vid_path):
        paths.append(vid_path)
    else:
        for file in vid_path.glob("*.MP4"):
            paths.append(file)
    for path in paths:
        frames, indices = video_to_frames(path, dump_frames=True)
        name = path.parent.name + "_" + path.name
        if name not in my_db.keys() or force_annotate:
            data = predict.predict_keypoints_locations(frames, args, name,
                                                       is_puppet=False,
                                                       save_intermed=False,
                                                       preloaded_model=my_model)
            my_db.setdefault(name, []).append({"data": data,
                                               "label": np.array([0, 0, 0]),
                                               "frame_indices": indices})
    if dump_to_db:
        file_io.dump_full_db(my_db, db_path)
    return my_db
