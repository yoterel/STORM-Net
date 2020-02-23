from pathlib import Path
import video
import tkinter as tk
from tkinter import filedialog
import pickle
import predict
from PIL import Image, ImageTk
import numpy as np

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


def process_video(vid_path, starting_frame=0, force_reselect=False):
    pickle_path = Path.joinpath(Path("data"), vid_path.name+"_frames.pickle")
    if pickle_path.is_file() and not force_reselect:
        f = open(pickle_path, 'rb')
        frames, indices = pickle.load(f)
        f.close()
    else:
        frames, indices = video.select_frames(vid_path, steps_per_datapoint=10, starting_frame=starting_frame)
        f = open(pickle_path, 'wb')
        pickle.dump([frames, indices], f)
        f.close()
    return frames, indices


def save_full_db(db):
    pickle_path = Path.joinpath(Path("data"), "full_db.pickle")
    f = open(pickle_path, 'wb')
    pickle.dump(db, f)
    f.close()


def load_full_db():
    pickle_path = Path.joinpath(Path("data"), "full_db.pickle")
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


def annotate_videos():  # contains GUI mainloop
    global new_db, frames
    root_videos_folder = Path("E:/University/masters/CapTracking/videos")
    base_model_folder = Path.joinpath(root_videos_folder, "openPos46")
    video_folder = Path.joinpath(root_videos_folder, "openPos47")
    paths = []
    for file in video_folder.glob("*.MP4"):
        paths.append(file)
    frames, indices = process_video(paths[video_number])
    new_db = load_full_db()
    if paths[video_number].name not in new_db.keys():
        data = predict.predict_keypoints_locations(frames, paths[video_number].name)
        new_db[paths[video_number].name] = {"data": data,
                                            "label": np.array([0, 0, 0]),
                                            "frame_indices": indices}
    current_video = paths[video_number].name
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
        frames, indices = process_video(paths[video_number], current_starting_frame_index+100, True)
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
    frame2 = tk.Frame(frame1, bg="white")
    frame2.pack(side="bottom")
    updateLabels()
    root.mainloop()
    return new_db

new_db = annotate_videos()
save_full_db(new_db)
