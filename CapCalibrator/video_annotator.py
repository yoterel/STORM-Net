from pathlib import Path
import video
import tkinter as tk
from tkinter import filedialog
import pickle
import predict
from PIL import Image, ImageTk
import numpy as np
import geometry
import argparse
import utils

### GLOBALS FOR GUI ###
video_number = 0
sticker_number = 0
frame_number = 0
data_points = 1
steps_per_datapoint = 10
number_of_features = 14
new_db = {}
frames =[]
### GLOBALS FOR GUI ###


def process_video(vid_path, dump_frames=False, starting_frame=0, force_reselect=False):
    pickle_path = Path.joinpath(Path("data"), vid_path.name+"_frames.pickle")
    if pickle_path.is_file() and not force_reselect:
        f = open(pickle_path, 'rb')
        frames, indices = pickle.load(f)
        f.close()
    else:
        frames, indices = video.select_frames(vid_path, steps_per_datapoint=10, starting_frame=starting_frame)
        if dump_frames:
            f = open(pickle_path, 'wb')
            pickle.dump([frames, indices], f)
            f.close()
    return frames, indices


def save_full_db(db, path=None):
    if path:
        pickle_path = path
    else:
        pickle_path = Path.joinpath(Path("data"), "full_db.pickle")
    f = open(pickle_path, 'wb')
    pickle.dump(db, f)
    f.close()


def load_full_db(db_path=None):
    if db_path is None:
        pickle_path = Path.joinpath(Path("data"), "full_db.pickle")
    else:
        pickle_path = db_path
    if pickle_path.is_file():
        f = open(pickle_path, 'rb')
        db = pickle.load(f)
        f.close()
    else:
        db = {}
        f = open(pickle_path, 'wb')
        pickle.dump(db, f)
        f.close()
    return db


def auto_annotate_videos(vid_folder, gt_digi_file):

    model_dir = Path("models")
    data_dir = Path("data")
    db_path = Path.joinpath(data_dir, "full_db.pickle")
    model_name = 'unet_try_2'
    model_full_name = Path.joinpath(model_dir, "{}_best_weights.h5".format(model_name))
    my_model = utils.load_semantic_seg_model(str(model_full_name))
    # get label
    my_digi_file = vid_folder.glob("*.txt").__next__()
    names, data = geometry.get_data_from_model_file(my_digi_file)
    sticker_data = geometry.get_sticker_data(names, data)
    names, data = geometry.get_data_from_model_file(gt_digi_file)
    gt_sticker_data = geometry.get_sticker_data(names, data)
    label = geometry.get_euler_angles(gt_sticker_data, sticker_data)  # obtain the angels needed to turn gt into my data
    label = np.array([1, -1, -1])*label  # flip y and z for network label
    # get data
    paths = []
    for file in vid_folder.glob("*.MP4"):
        paths.append(file)
    my_db = load_full_db(db_path)
    for path in paths:
        print("processing video:", path)
        if path.name not in my_db.keys():
            frames, indices = process_video(path)
            data = predict.predict_keypoints_locations(frames, path.name, True, False, my_model, 1)
            my_db[path.name] = {"data": data,
                                 "label": np.array(label),
                                 "frame_indices": indices}
            save_full_db(my_db, db_path)
    return my_db


def annotate_videos(video_folder):  # contains GUI mainloop
    global new_db, frames
    paths = []
    for file in video_folder.glob("*.MP4"):
        paths.append(file)
    frames, indices = process_video(paths[video_number])
    new_db = load_full_db()
    if paths[video_number].name not in new_db.keys():
        data = predict.predict_keypoints_locations(frames, paths[video_number].name, True, False)
        new_db[paths[video_number].name] = {"data": data,
                                            "label": np.array([0, 0, 0]),
                                            "frame_indices": indices}
    root = tk.Tk()
    root.title("Video Annotator")
    root.resizable(False, False)
    root.bind("<Escape>", lambda e: root.destroy())
    root.configure(background='white')
    def saveCoords(event):
        global sticker_number, new_db
        current_video = paths[video_number].name
        new_db[current_video]["data"][0, frame_number, sticker_number:sticker_number+2] = event.x, 540-event.y
        if sticker_number >= 12:
            sticker_number = 0
        else:
            sticker_number += 2
        updateLabels()
    def zeroCoords(event):
        global sticker_number, new_db
        current_video = paths[video_number].name
        new_db[current_video]["data"][0, frame_number, sticker_number:sticker_number+2] = 0, 0
        if sticker_number >= 12:
            sticker_number = 0
        else:
            sticker_number += 2
        updateLabels()
    def nextCoords(event):
        global sticker_number, new_db
        if sticker_number >= 12:
            sticker_number = 0
        else:
            sticker_number += 2
        updateLabels()
    def updateLabels():
        global new_db
        clearLabels()
        current_video = paths[video_number].name
        db_to_show = np.reshape(new_db[current_video]["data"][0, frame_number, :], (7, 2))
        sticker_names = ["AL", "NZ", "AR", "CAP1", "CAP2", "CAP3", "CAP4"]
        my_string = "File Name: {}".format(current_video)
        label = tk.Label(frame2, text=my_string, bg="white")
        label.pack(fill="x")
        my_string = "Frame: {}".format(frame_number)
        label = tk.Label(frame2, text=my_string, bg="white")
        label.pack(fill="x")
        for i in range(len(db_to_show)):
            my_string = "{}: {},{}".format(sticker_names[i], int(db_to_show[i, 0]), int(db_to_show[i, 1]))
            if i == sticker_number // 2:
                label = tk.Label(frame2, text=my_string, bg="gray")
            else:
                label = tk.Label(frame2, text=my_string, bg="white")
            label.pack(fill="x")
            canvas.create_line(int(db_to_show[i, 0]) - 5, 540-(int(db_to_show[i, 1])) - 5, int(db_to_show[i, 0]) + 5,
                               540-(int(db_to_show[i, 1])) + 5, fill="red", tag="cross")
            canvas.create_line(int(db_to_show[i, 0]) + 5, 540 - (int(db_to_show[i, 1])) - 5, int(db_to_show[i, 0]) - 5,
                               540-(int(db_to_show[i, 1])) + 5, fill="red", tag="cross")
    def nextVideo():
        global frames, new_db, video_number, frame_number, sticker_number
        if video_number < (len(paths)-1):
            video_number += 1
            frames, indices = process_video(paths[video_number])
            if paths[video_number].name not in new_db.keys():
                data = predict.predict_keypoints_locations(frames, is_puppet=True, vid_name=paths[video_number].name)
                new_db[paths[video_number].name] = {"data": data,
                                                    "label": np.array([0, 0, 0]),
                                                    "frame_indices": indices}
            frame_number = 0
            sticker_number = 0
            canvas.delete("image")
            canvas.delete("cross")
            img = ImageTk.PhotoImage(image=frames[frame_number])
            canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
            canvas.image = img  # keep a reference or it gets deleted
            updateLabels()
    def prevVideo():
        global frames, new_db, video_number, frame_number, sticker_number
        if video_number > 0:
            video_number -= 1
            frames, indices = process_video(paths[video_number])
            if paths[video_number].name not in new_db.keys():
                data = predict.predict_keypoints_locations(frames, is_puppet=True, vid_name=paths[video_number].name)
                new_db[paths[video_number].name] = {"data": data,
                                                    "label": np.array([0, 0, 0]),
                                                    "frame_indices": indices}
            frame_number = 0
            sticker_number = 0
            canvas.delete("image")
            canvas.delete("cross")
            img = ImageTk.PhotoImage(image=frames[frame_number])
            canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
            canvas.image = img  # keep a reference or it gets deleted
            updateLabels()
    def reselectFrames():
        global frames, new_db, frame_number, sticker_number
        current_video = paths[video_number].name
        current_starting_frame_index = new_db[current_video]["frame_indices"][0]
        frames, indices = process_video(paths[video_number], False, current_starting_frame_index+100, True)
        new_db[current_video]["frame_indices"] = indices
        frame_number = 0
        sticker_number = 0
        canvas.delete("image")
        canvas.delete("cross")
        img = ImageTk.PhotoImage(image=frames[frame_number])
        canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
        canvas.image = img  # keep a reference or it gets deleted
        updateLabels()
    def nextFrame():
        global frame_number, sticker_number
        if frame_number < 9:
            frame_number += 1
            sticker_number = 0
            canvas.delete("image")
            canvas.delete("cross")
            img = ImageTk.PhotoImage(image=frames[frame_number])
            canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
            canvas.image = img  # keep a reference or it gets deleted
            updateLabels()
    def prevFrame():
        global frame_number, sticker_number
        if frame_number > 0:
            frame_number -= 1
            sticker_number = 0
            canvas.delete("image")
            img = ImageTk.PhotoImage(image=frames[frame_number])
            canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
            canvas.image = img  # keep a reference or it gets deleted
            updateLabels()
    def loadSession():
        global new_db
        clearLabels()
        filename = filedialog.askopenfilename(initialdir="./data", title="Select Session File")
        f = open(filename, 'rb')
        new_db = pickle.load(f)
        if paths[video_number].name not in new_db.keys():
            data = predict.predict_keypoints_locations(frames, paths[video_number].name)
            new_db[paths[video_number].name] = {"data": data,
                                                "label": np.array([0, 0, 0]),
                                                "frame_indices": indices}
        f.close()
        updateLabels()
    def saveSession():
        global new_db
        f = filedialog.asksaveasfile(initialdir="./data", title="Select Session File", mode='wb', initialfile="full_db.pickle")
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        pickle.dump(new_db, f)
        f.close()
    def clearLabels():
        canvas.delete("cross")
        for widget in frame2.winfo_children():
            widget.destroy()
    def test():
        global new_db
        data = predict.predict_keypoints_locations(frames, paths[video_number].name, True, False)
        new_db[paths[video_number].name] = {"data": data,
                                            "label": np.array([0, 0, 0]),
                                            "frame_indices": indices}
        updateLabels()
    canvas = tk.Canvas(root, height=frames[0].size[1], width=frames[0].size[0], bg="#263D42")
    canvas.pack(side="left")
    img = ImageTk.PhotoImage(image=frames[0])
    canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
    # canvas.image = img  # keep a reference!
    canvas.bind("<ButtonPress-1>", saveCoords)
    canvas.bind("<ButtonPress-2>", zeroCoords)
    canvas.bind("<ButtonPress-3>", nextCoords)
    frame1 = tk.Frame(root, bg="white")
    frame1.pack(anchor="ne", side="right")
    nextFrame = tk.Button(frame1, text="Next Frame", padx=10, pady=5, fg="white", bg="#263D42", command=nextFrame)
    nextFrame.pack(fill="x")
    prevFrame = tk.Button(frame1, text="Previous Frame", padx=10, pady=5, fg="white", bg="#263D42", command=prevFrame)
    prevFrame.pack(fill="x")
    nextVideo = tk.Button(frame1, text="Next Video", padx=10, pady=5, fg="white", bg="#263D42", command=nextVideo)
    nextVideo.pack(fill="x")
    prevVideo = tk.Button(frame1, text="Previous Video", padx=10, pady=5, fg="white", bg="#263D42", command=prevVideo)
    prevVideo.pack(fill="x")
    reselectFrames = tk.Button(frame1, text="Reselect Frames", padx=10, pady=5, fg="white", bg="#263D42", command=reselectFrames)
    reselectFrames.pack(fill="x")
    loadSession = tk.Button(frame1, text="Load Session", padx=10, pady=5, fg="white", bg="#263D42", command=loadSession)
    loadSession.pack(fill="x")
    saveSession = tk.Button(frame1, text="Save Session", padx=10, pady=5, fg="white", bg="#263D42", command=saveSession)
    saveSession.pack(fill="x")
    doneButton = tk.Button(frame1, text="Done", padx=10, pady=5, fg="white", bg="#263D42", command=root.destroy)
    doneButton.pack(fill="x")
    testButton = tk.Button(frame1, text="Test", padx=10, pady=5, fg="white", bg="#263D42", command=test)
    testButton.pack(fill="x")
    frame2 = tk.Frame(frame1, bg="white")
    frame2.pack(side="bottom")
    updateLabels()
    root.mainloop()
    return new_db


def parse_arguments():
    parser = argparse.ArgumentParser(description='Automatically annotates FNIRS videos on disk.')
    parser.add_argument("video_folder", help="The path to the video folder.")
    parser.add_argument("model_file", help="The base model file path.")
    parser.add_argument("-g", "--gui", action='store_true', help="Shows GUI")
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    # cmd_line = '/disk1/yotam/capnet/openPos/openPos/openPos49/ /disk1/yotam/capnet/openPos/openPos/openPos46/ '.split()
    # cmd_line = 'E:/University/masters/CapTracking/videos/openPos49 E:/University/masters/CapTracking/videos/openPos46 -g'.split()
    args = parser.parse_args()  # cmd_line
    args.video_folder = Path(args.video_folder)
    args.model_file = Path(args.model_file)
    if Path.is_dir(args.model_file):
        args.model_file = args.model_file.glob("*.txt").__next__()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    new_db = auto_annotate_videos(args.video_folder, args.model_file)
    if args.gui:
        annotate_videos(args.video_folder)
