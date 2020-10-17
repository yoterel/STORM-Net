from pathlib import Path
import video
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import predict
from PIL import Image, ImageTk
import numpy as np
import argparse
import file_io
import queue
import threading
import time


### GLOBALS FOR GUI ###
video_number = 0
sticker_number = 0
frame_number = 0
shift = 0
steps_per_datapoint = 10
number_of_features = 14
new_db = {}
frames =[]
### GLOBALS FOR GUI ###


class AnnotationPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        h, w = self.controller.get_frame_size()
        self.canvas = tk.Canvas(self, height=h, width=w, bg="#263D42")
        img = ImageTk.PhotoImage(master=self, image=self.controller.get_cur_frame())
        self.canvas.img = img  # or else image gets garbage collected
        self.canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
        self.canvas.bind("<ButtonPress-1>", self.controller.saveCoords)
        self.canvas.bind("<ButtonPress-2>", self.controller.zeroCoords)
        self.canvas.bind("<ButtonPress-3>", self.controller.nextCoords)
        pad_x = 0
        pad_y = 5
        button_width = 5
        nextFrame = tk.Button(self, text="Next Frame", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                              bg="#263D42", command=self.controller.nextFrame)
        prevFrame = tk.Button(self, text="Previous Frame", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                              bg="#263D42", command=self.controller.prevFrame)
        nextVideo = tk.Button(self, text="Next Video", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                              bg="#263D42", command=self.controller.nextVideo)
        prevVideo = tk.Button(self, text="Previous Video", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                              bg="#263D42", command=self.controller.prevVideo)
        shiftVideoF = tk.Button(self, text="Next Shift", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                                bg="#263D42", command=self.controller.shiftVideoF)
        shiftVideoB = tk.Button(self, text="Prev Shift", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                                bg="#263D42", command=self.controller.shiftVideoB)
        loadSession = tk.Button(self, text="Load Session", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                                bg="#263D42", command=self.controller.loadSession)
        saveSession = tk.Button(self, text="Save Session", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                                bg="#263D42", command=self.controller.saveSession)
        doneButton = tk.Button(self, text="Done", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                               bg="#263D42", command=self.controller.destroy)
        self.data_panel = tk.Frame(self, bg="white")
        self.update_labels()
        loadSession.grid(row=0, column=0, sticky="w"+"e")
        saveSession.grid(row=0, column=1, sticky="w"+"e")
        nextVideo.grid(row=0, column=2, sticky="w"+"e")
        prevVideo.grid(row=0, column=3, sticky="w"+"e")
        nextFrame.grid(row=0, column=4, sticky="w"+"e")
        prevFrame.grid(row=0, column=5, sticky="w"+"e")
        shiftVideoF.grid(row=0, column=6, sticky="w"+"e")
        shiftVideoB.grid(row=0, column=7, sticky="w"+"e")
        doneButton.grid(row=0, column=8, sticky="w"+"e")
        self.data_panel.grid(row=0, column=9, rowspan=9)
        self.canvas.grid(row=1, columnspan=9, rowspan=8)

    def update_labels(self):
        self.clear_labels()
        db = self.controller.get_db()
        cur_frame_index = self.controller.get_cur_frame_index()
        cur_sticker_index = self.controller.get_cur_sticker_index()
        shift = 0
        pad_y = 5
        cur_video_name = self.controller.get_cur_video_name()
        db_to_show = np.reshape(db[cur_video_name][shift]["data"][0, cur_frame_index, :], (7, 2))
        sticker_names = ["Left Eye", "Nose Tip", "Right Eye", "CAP1", "CAP2", "CAP3", "CAP4"]
        my_string = "File Name: {}".format(cur_video_name)
        label = tk.Label(self.data_panel, text=my_string, width=30, bg="white", anchor="center", pady=pad_y)
        label.pack(fill="x")
        my_string = "Frame: {}".format(frame_number)
        label = tk.Label(self.data_panel, text=my_string, width=30, bg="white", anchor="center", pady=pad_y)
        label.pack(fill="x")
        for i in range(7):
            my_string = "{}: {},{}".format(sticker_names[i], int(db_to_show[i, 0]), int(db_to_show[i, 1]))
            if i == cur_sticker_index // 2:
                sticker_label = tk.Label(self.data_panel, text=my_string, bg="gray", width=15, anchor="w", pady=pad_y)
            else:
                sticker_label = tk.Label(self.data_panel, text=my_string, bg="white", width=15, anchor="w", pady=pad_y)
            sticker_label.pack(fill="x")
            self.canvas.create_line(int(db_to_show[i, 0]) - 5, 540 - (int(db_to_show[i, 1])) - 5, int(db_to_show[i, 0]) + 5,
                               540 - (int(db_to_show[i, 1])) + 5, fill="red", tag="cross")
            self.canvas.create_line(int(db_to_show[i, 0]) + 5, 540 - (int(db_to_show[i, 1])) - 5, int(db_to_show[i, 0]) - 5,
                               540 - (int(db_to_show[i, 1])) + 5, fill="red", tag="cross")

    def update_canvas(self):
        self.canvas.delete("image")
        self.canvas.delete("cross")
        img = ImageTk.PhotoImage(master=self.canvas, image=self.controller.get_cur_frame())
        self.canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
        self.canvas.image = img  # keep a reference or it gets deleted

    def clear_labels(self):
        self.canvas.delete("cross")
        for widget in self.data_panel.winfo_children():
            widget.destroy()


class ProgressBarPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.prog_bar = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=300, mode="indeterminate")
        self.prog_bar.pack(fill="none", expand=True)

    def show_progress(self, show):
        if show:
            self.prog_bar.start(10)
        else:
            self.prog_bar.stop()


class GUI(tk.Tk):
    def __init__(self, db, paths, mode):
        super().__init__()
        self.db = db
        self.paths = paths
        self.mode = mode
        self.shift = 0
        self.cur_video_index = 0
        self.cur_frame_index = 0
        self.cur_sticker_index = 0
        self.frames = None
        self.indices = None
        self.queue = queue.Queue()

        self.wm_title("Video Annotator")
        self.resizable(False, False)
        self.bind("<Escape>", lambda e: self.destroy())
        self.configure(background='white')
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.panels = {}
        for F in (AnnotationPage, ProgressBarPage):
            panel = F(self.container, self)
            self.panels[F] = panel
            panel.grid(row=0, column=0, sticky="nsew")
        self.show_panel(ProgressBarPage)
        self.take_async_action(["video_to_frames", self.paths, self.cur_video_index, self.mode, self.db])

    def process_queue(self):
        try:
            msg = self.queue.get(False)
            if msg[0] == "video_to_frames":
                self.frames, self.indices, my_dict = msg[1:]
                if my_dict:
                    self.db.setdefault(self.get_cur_video_name(), []).append(my_dict)
                self.panels[AnnotationPage].update_canvas()
                self.panels[AnnotationPage].update_labels()
            elif msg[0] == "shift_video":
                self.frames, self.indices, self.db[self.cur_video_index][self.shift]["frame_indices"] = msg[1:]
                print("new indices:", self.indices)
                self.panels[AnnotationPage].update_canvas()
                self.panels[AnnotationPage].update_labels()
            # Show result of the task if needed
            self.panels[ProgressBarPage].show_progress(False)
            self.show_panel(AnnotationPage)
            self.enable_menu()
        except queue.Empty:
            self.after(100, self.process_queue)

    def take_async_action(self, msg_id):
        self.disbale_menu()
        self.show_panel(ProgressBarPage)
        self.panels[ProgressBarPage].show_progress(True)
        ThreadedTask(self.queue, msg_id).start()
        self.after(100, self.process_queue)

    def show_panel(self, cont):
        panel = self.panels[cont]
        panel.tkraise()

    def get_frame_size(self):
        """
        returns frame size
        :return: h, w
        """
        if self.frames:
            return self.frames[0].size[1], self.frames[0].size[0]
        else:
            return 540, 960

    def get_db(self):
        return self.db

    def get_cur_frame_index(self):
        return self.cur_frame_index

    def get_cur_sticker_index(self):
        return self.cur_sticker_index

    def get_paths(self):
        return self.paths

    def get_cur_frame(self):
        if self.frames:
            return self.frames[self.cur_frame_index]
        else:
            return Image.fromarray(np.zeros(self.get_frame_size()))

    def get_cur_video_name(self):
        return self.paths[self.cur_video_index].parent.name + "_" + self.paths[self.cur_video_index].name


    def disbale_menu(self):
        for w in self.container.winfo_children():
            if w.winfo_class() == "Button":
                w.configure(state="disabled")

    def enable_menu(self):
        for w in self.container.winfo_children():
            if w.winfo_class() == "Button":
                w.configure(state="normal")

    def go_to_next_coord(self):
        if self.cur_sticker_index >= 12:
            self.cur_sticker_index = 0
        else:
            self.cur_sticker_index += 2
        self.panels[AnnotationPage].update_labels()

    def saveCoords(self, event):
        current_video = self.get_cur_video_name()
        self.db[current_video][self.shift]["data"][0, self.cur_frame_index, self.cur_sticker_index:self.cur_sticker_index+2] = event.x, 540-event.y
        self.go_to_next_coord()
        self.updateLabels()

    def zeroCoords(self):
        current_video = self.get_cur_video_name()
        self.db[current_video][self.shift]["data"][0, self.cur_frame_index, self.cur_sticker_index:self.cur_sticker_index+2] = 0, 0
        self.go_to_next_coord()
        self.updateLabels()

    def nextCoords(self):
        self.go_to_next_coord()
        self.updateLabels()

    def nextVideo(self):
        if self.cur_video_index < (len(self.paths)-1):
            self.cur_video_index += 1
            self.cur_frame_index = 0
            self.cur_sticker_index = 0
            self.take_async_action(["video_to_frames", self.paths, self.cur_video_index, self.mode, self.db])

    def prevVideo(self):
        if self.cur_video_index > 0:
            self.cur_video_index -= 1
            self.cur_frame_index = 0
            self.cur_sticker_index = 0
            self.take_async_action(["video_to_frames", self.paths, self.cur_video_index, self.mode, self.db])

    def shiftVideoF(self):
        current_video = self.get_cur_video_name()
        current_indices = self.db[current_video][shift]["frame_indices"]
        print("current indices:", current_indices)
        new_indices = current_indices.copy()
        new_indices[frame_number] += 1
        self.take_async_action(["shift_video", self.paths, self.cur_video_index, new_indices])

    def shiftVideoB(self):
        current_video = self.get_cur_video_name()
        current_indices = self.db[current_video][shift]["frame_indices"]
        print("current indices:", current_indices)
        new_indices = current_indices.copy()
        new_indices[frame_number] -= 1
        self.take_async_action(["shift_video", self.paths, self.cur_video_index, new_indices])

    def nextFrame(self):
        if self.cur_frame_index < 9:
            self.cur_frame_index += 1
            self.cur_sticker_index = 0
            self.panels[AnnotationPage].update_canvas()
            self.panels[AnnotationPage].update_labels()

    def prevFrame(self):
        if self.cur_frame_index > 0:
            self.cur_frame_index -= 1
            self.cur_sticker_index = 0
            self.panels[AnnotationPage].update_canvas()
            self.panels[AnnotationPage].update_labels()

    def loadSession(self):
        global new_db
        self.clearLabels()
        filename = filedialog.askopenfilename(initialdir="./data", title="Select Session File")
        f = open(filename, 'rb')
        new_db = pickle.load(f)
        name = paths[video_number].parent.name + "_" + paths[video_number].name
        if name not in new_db.keys():
            data = predict.predict_keypoints_locations(frames, name)
            new_db.setdefault(name, []).append({"data": data,
                                                "label": np.array([0, 0, 0]),
                                                "frame_indices": indices})
        f.close()
        self.updateLabels()

    def saveSession(self):
        global new_db
        f = filedialog.asksaveasfile(initialdir="./data", title="Select Session File", mode='wb')
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        pickle.dump(new_db, f)
        f.close()


class ThreadedTask(threading.Thread):
    def __init__(self, queue, msg):
        threading.Thread.__init__(self)
        self.queue = queue
        self.msg = msg

    def run(self):
        if self.msg[0] == "video_to_frames":
            self.handle_video_to_frames()
        if self.msg[0] == "shift_video":
            self.handle_shift_video()

    def handle_video_to_frames(self):
        paths, cur_video_index, mode, db = self.msg[1:]
        my_dict = {}
        frames, indices = video.video_to_frames(paths[cur_video_index], dump_frames=True)
        name = paths[cur_video_index].parent.name + "_" + paths[cur_video_index].name
        if name not in db.keys():
            if mode == "semi-auto" or mode == "special":
                data = predict.predict_keypoints_locations(frames, name, is_puppet=False, save_intermed=False)
            else:
                data = np.zeros((1, 10, 14))
            my_dict = {"data": data,
                       "label": np.array([0, 0, 0]),
                       "frame_indices": indices}
        self.queue.put(["video_to_frames", frames, indices, my_dict])

    def handle_shift_video(self):
        paths, cur_video_index, new_indices = self.msg[1:]
        frames, indices = video.video_to_frames(paths[cur_video_index],
                                                dump_frames=True,
                                                frame_indices=new_indices,
                                                force_reselect=True)
        self.queue.put(["shift_video", frames, indices])


def annotate_videos(video_path, mode="auto", v=0):  # contains GUI mainloop
    # global new_db, frames, video_number
    if mode == "special":
        special_db = Path.joinpath(Path("data"), "telaviv_db.pickle")
        new_db = file_io.load_full_db(special_db)
    else:
        new_db = file_io.load_full_db()
    paths = []
    if Path.is_file(video_path):
        paths.append(video_path)
        video_name = video_path.name
        video_number = [x.name for x in paths].index(video_name)
    else:
        for file in video_path.glob("*.MP4"):
            paths.append(file)

    app = GUI(new_db, paths, mode)
    app.mainloop()
    print("test")

    root = tk.Tk()
    root.title("Video Annotator")
    root.resizable(False, False)
    root.bind("<Escape>", lambda e: root.destroy())
    root.configure(background='white')


    def saveCoords(event):
        global sticker_number, new_db
        current_video = paths[video_number].parent.name + "_" + paths[video_number].name
        new_db[current_video][shift]["data"][0, frame_number, sticker_number:sticker_number+2] = event.x, 540-event.y
        if sticker_number >= 12:
            sticker_number = 0
        else:
            sticker_number += 2
        updateLabels()

    def zeroCoords(event):
        global sticker_number, new_db
        current_video = paths[video_number].parent.name + "_" + paths[video_number].name
        new_db[current_video][shift]["data"][0, frame_number, sticker_number:sticker_number+2] = 0, 0
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
        current_video = paths[video_number].parent.name + "_" + paths[video_number].name
        db_to_show = np.reshape(new_db[current_video][shift]["data"][0, frame_number, :], (7, 2))
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
            frames, indices = video.video_to_frames(paths[video_number], dump_frames=True)
            name = paths[video_number].parent.name + "_" + paths[video_number].name
            if name not in new_db.keys():
                if mode == "semi-auto" or mode == "auto" or mode == "special":
                    data = predict.predict_keypoints_locations(frames, is_puppet=False, vid_name=name)
                else:
                    data = np.zeros((1, 10, 14))
                new_db.setdefault(name, []).append({"data": data,
                                                    "label": np.array([0, 0, 0]),
                                                    "frame_indices": indices})
            frame_number = 0
            sticker_number = 0
            canvas.delete("image")
            canvas.delete("cross")
            img = ImageTk.PhotoImage(master=canvas, image=frames[frame_number])
            canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
            canvas.image = img  # keep a reference or it gets deleted
            updateLabels()

    def prevVideo():
        global frames, new_db, video_number, frame_number, sticker_number
        if video_number > 0:
            video_number -= 1
            frames, indices = video.video_to_frames(paths[video_number], dump_frames=True)
            name = paths[video_number].parent.name + "_" + paths[video_number].name
            if name not in new_db.keys():
                if mode == "semi-auto" or mode == "auto" or mode=="special":
                    data = predict.predict_keypoints_locations(frames, is_puppet=False, vid_name=name)
                else:
                    data = np.zeros((1, 10, 14))
                new_db.setdefault(name, []).append({"data": data,
                                                    "label": np.array([0, 0, 0]),
                                                    "frame_indices": indices})
            frame_number = 0
            sticker_number = 0
            canvas.delete("image")
            canvas.delete("cross")
            img = ImageTk.PhotoImage(master=canvas, image=frames[frame_number])
            canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
            canvas.image = img  # keep a reference or it gets deleted
            updateLabels()

    def reselectFrames():
        global frames, new_db, frame_number, sticker_number
        current_video = paths[video_number].parent.name + "_" + paths[video_number].name
        current_starting_frame_index = new_db[current_video][shift]["frame_indices"][0]
        frames, indices = video.video_to_frames(paths[video_number],
                                        dump_frames=True,
                                        starting_frame=current_starting_frame_index+60,
                                        force_reselect=True)
        new_db[current_video][shift]["frame_indices"] = indices
        frame_number = 0
        sticker_number = 0
        canvas.delete("image")
        canvas.delete("cross")
        img = ImageTk.PhotoImage(master=canvas, image=frames[frame_number])
        canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
        canvas.image = img  # keep a reference or it gets deleted
        updateLabels()

    def shiftVideoF():
        global new_db, frames, frame_number, sticker_number
        current_video = paths[video_number].parent.name + "_" + paths[video_number].name
        current_indices = new_db[current_video][shift]["frame_indices"]
        print("current indices:", current_indices)
        new_indices = current_indices.copy()
        new_indices[frame_number] += 1
        frames, indices = video.video_to_frames(paths[video_number],
                                        dump_frames=True,
                                        frame_indices=new_indices,
                                        force_reselect=True)
        new_db[current_video][shift]["frame_indices"] = new_indices
        print("new indices:", new_indices)
        canvas.delete("image")
        canvas.delete("cross")
        img = ImageTk.PhotoImage(master=canvas, image=frames[frame_number])
        canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
        canvas.image = img  # keep a reference or it gets deleted
        updateLabels()

    def shiftVideoB():
        global new_db, frames, frame_number, sticker_number
        current_video = paths[video_number].parent.name + "_" + paths[video_number].name
        current_indices = new_db[current_video][shift]["frame_indices"]
        print("current indices:", current_indices)
        new_indices = current_indices.copy()
        new_indices[frame_number] -= 1
        frames, indices = video.video_to_frames(paths[video_number],
                                        dump_frames=True,
                                        frame_indices=new_indices,
                                        force_reselect=True)
        new_db[current_video][shift]["frame_indices"] = new_indices
        print("new indices:", new_indices)
        canvas.delete("image")
        canvas.delete("cross")
        img = ImageTk.PhotoImage(master=canvas, image=frames[frame_number])
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
            img = ImageTk.PhotoImage(master=canvas, image=frames[frame_number])
            canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
            canvas.image = img  # keep a reference or it gets deleted
            updateLabels()

    def prevFrame():
        global frame_number, sticker_number
        if frame_number > 0:
            frame_number -= 1
            sticker_number = 0
            canvas.delete("image")
            img = ImageTk.PhotoImage(master=canvas, image=frames[frame_number])
            canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
            canvas.image = img  # keep a reference or it gets deleted
            updateLabels()

    def loadSession():
        global new_db
        clearLabels()
        filename = filedialog.askopenfilename(initialdir="./data", title="Select Session File")
        f = open(filename, 'rb')
        new_db = pickle.load(f)
        name = paths[video_number].parent.name + "_" + paths[video_number].name
        if name not in new_db.keys():
            data = predict.predict_keypoints_locations(frames, name)
            new_db.setdefault(name, []).append({"data": data,
                                                "label": np.array([0, 0, 0]),
                                                "frame_indices": indices})
        f.close()
        updateLabels()

    def saveSession():
        global new_db
        f = filedialog.asksaveasfile(initialdir="./data", title="Select Session File", mode='wb')
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        pickle.dump(new_db, f)
        f.close()

    def clearLabels():
        canvas.delete("cross")
        for widget in frame2.winfo_children():
            widget.destroy()
    canvas = tk.Canvas(root, height=frames[0].size[1], width=frames[0].size[0], bg="#263D42")

    img = ImageTk.PhotoImage(master=canvas, image=frames[0])
    canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
    # canvas.image = img  # keep a reference!
    canvas.bind("<ButtonPress-1>", saveCoords)
    canvas.bind("<ButtonPress-2>", zeroCoords)
    canvas.bind("<ButtonPress-3>", nextCoords)
    frame1 = tk.Frame(root, bg="white")
    nextFrame = tk.Button(frame1, text="Next Frame", padx=10, pady=5, fg="white", bg="#263D42", command=nextFrame)
    prevFrame = tk.Button(frame1, text="Previous Frame", padx=10, pady=5, fg="white", bg="#263D42", command=prevFrame)
    nextVideo = tk.Button(frame1, text="Next Video", padx=10, pady=5, fg="white", bg="#263D42", command=nextVideo)
    prevVideo = tk.Button(frame1, text="Previous Video", padx=10, pady=5, fg="white", bg="#263D42", command=prevVideo)
    reselectFrames = tk.Button(frame1, text="Reselect Frames", padx=10, pady=5, fg="white", bg="#263D42", command=reselectFrames)
    shiftVideoF = tk.Button(frame1, text="Next Shift", padx=10, pady=5, fg="white", bg="#263D42", command=shiftVideoF)
    shiftVideoB = tk.Button(frame1, text="Prev Shift", padx=10, pady=5, fg="white", bg="#263D42", command=shiftVideoB)
    loadSession = tk.Button(frame1, text="Load Session", padx=10, pady=5, fg="white", bg="#263D42", command=loadSession)
    saveSession = tk.Button(frame1, text="Save Session", padx=10, pady=5, fg="white", bg="#263D42", command=saveSession)
    doneButton = tk.Button(frame1, text="Done", padx=10, pady=5, fg="white", bg="#263D42", command=root.destroy)
    frame2 = tk.Frame(frame1, bg="white")

    canvas.pack(side="left")
    frame1.pack(anchor="ne", side="right")
    nextFrame.pack(fill="x")
    prevFrame.pack(fill="x")
    nextVideo.pack(fill="x")
    prevVideo.pack(fill="x")
    reselectFrames.pack(fill="x")
    shiftVideoF.pack(fill="x")
    shiftVideoB.pack(fill="x")
    loadSession.pack(fill="x")
    saveSession.pack(fill="x")
    doneButton.pack(fill="x")
    frame2.pack(side="bottom")
    updateLabels()
    root.mainloop()

    return new_db


def parse_arguments():
    parser = argparse.ArgumentParser(description='Automatically annotates FNIRS videos on disk.')
    parser.add_argument("video_folder", help="The path to the video folder.")
    parser.add_argument("model_file", help="The base model file path.")
    parser.add_argument("-a", "--auto_annotate", action='store_true', help="Automatically annotates videos in folder")
    parser.add_argument("-g", "--gui", action='store_true', help="Shows GUI")
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    # cmd_line = '/disk1/yotam/capnet/openPos/openPos/openPos49/ /disk1/yotam/capnet/openPos/openPos/openPos46/ '.split()
    cmd_line = 'E:/University/masters/CapTracking/videos/openPos53/GX011577.MP4 E:/University/masters/CapTracking/videos/openPos53 -g'.split()
    args = parser.parse_args(cmd_line)  # cmd_line
    args.video_folder = Path(args.video_folder)
    args.model_file = Path(args.model_file)
    if Path.is_dir(args.model_file):
        args.model_file = args.model_file.glob("*.txt").__next__()
    return args


if __name__ == "__main__":
    app = GUI()
    app.mainloop()

    # blacklist = ["GX011543.MP4", "GX011544.MP4", "GX011547.MP4", "GX011549.MP4",
    #              "GX011537.MP4", "GX011538.MP4"]
    # args = parse_arguments()
    # if args.auto_annotate:
    #     new_db = video.auto_annotate_videos(args.video_folder, args.model_file)
    # if args.gui:
    #     annotate_videos(args.video_folder)
