import cv2
import numpy as np
from pathlib import Path
import imageio
import tkinter as tk
from tkinter import filedialog
import pickle
from PIL import Image, ImageTk
import predict

### GLOBALS FOR GUI ###
sticker_number = 0
frame_number = 0
data_points = 1
steps_per_datapoint = 10
number_of_features = 14
new_db = np.zeros((data_points, steps_per_datapoint, number_of_features))
### GLOBALS FOR GUI ###


def process_video(vid_path, automation_level, v):
    pickle_path = Path.joinpath(Path("data"), vid_path.name+"_frames.pickle")
    if pickle_path.is_file():
        if v:
            print("Loading selected frames from:", pickle_path)
        f = open(pickle_path, 'rb')
        frames = pickle.load(f)
        f.close()
    else:
        if v:
            print("Selecting 10 frames from:", vid_path)
        frames = select_frames(vid_path, steps_per_datapoint)
        f = open(pickle_path, 'wb')
        pickle.dump(frames, f)
        f.close()
    return annotate_video(frames, automation_level, vid_path.name, v)


def select_frames(vid_path, steps_per_datapoint=10, starting_frame=0):

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
        if i >= frames_to_use:
            break
        else:
            if i % (frames_to_use // steps_per_datapoint) == starting_frame:
                frames.append(Image.fromarray(im).resize((960, 540)))
                indices.append(i)
    return frames, indices


def annotate_video(frames, automation_level, vid_name, v):  # contains GUI mainloop
    global new_db
    if automation_level == "semi-auto" or automation_level == "auto":
        if v:
            print("Predicting key points from frames.")
        new_db = predict.predict_keypoints_locations(frames, vid_name, v)
    if automation_level == "semi-auto" or automation_level == "manual":
        if v:
            print("Launching GUI to manually fix/annotate frames.")
        root = tk.Tk()
        root.title("Video Annotator: " + vid_name)
        root.resizable(False, False)
        root.bind("<Escape>", lambda e: root.destroy())
        root.configure(background='white')
        def saveCoords(event):
            global sticker_number, new_db
            # global frame_number, sticker_number, new_db
            new_db[0, frame_number, sticker_number:sticker_number+2] = event.x, 540-event.y
            if sticker_number >= 12:
                sticker_number = 0
            else:
                sticker_number += 2
            updateLabels()
        def zeroCoords(event):
            global sticker_number, new_db
            # global frame_number, sticker_number, new_db
            new_db[0, frame_number, sticker_number:sticker_number+2] = 0, 0
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
            db_to_show = np.reshape(new_db[0, frame_number, :], (7, 2))
            sticker_names = ["AL", "NZ", "AR", "CAP1", "CAP2", "CAP3", "CAP4"]
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
            f.close()
            updateLabels()
        def saveSession():
            global new_db
            f = filedialog.asksaveasfile(initialdir="./data", title="Select Session File", mode='wb', initialfile="my_session.pickle")
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
