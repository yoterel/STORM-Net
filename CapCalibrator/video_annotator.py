from pathlib import Path
import video
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, simpledialog, messagebox
import predict
from PIL import Image, ImageTk
import numpy as np
import file_io
import queue
import threading
import utils
import geometry


class CalibrationPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        h, w = self.controller.get_frame_size()
        self.canvas = tk.Canvas(self, height=h, width=w, highlightthickness=0, bg="#263D42")
        img = ImageTk.PhotoImage(master=self, image=self.controller.get_cur_frame())
        self.canvas.img = img  # or else image gets garbage collected
        self.canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
        self.canvas.bind("<ButtonPress-1>", self.controller.save_coords)
        self.canvas.bind("<ButtonPress-2>", self.controller.zero_coords)
        self.canvas.bind("<ButtonPress-3>", self.controller.next_coords)
        self.bind("<Left>", self.controller.prev_frame)
        self.bind("<Right>", self.controller.next_frame)
        self.data_panel = tk.Frame(self, bg="white")
        self.canvas.grid(row=0, column=0)
        self.data_panel.grid(row=0, column=1)

    def update_labels(self):
        self.clear_labels()
        shift = 0
        pad_y = 5

        db = self.controller.get_db()
        cur_frame_index = self.controller.get_cur_frame_index()
        cur_sticker_index = self.controller.get_cur_sticker_index()
        cur_video_name = self.controller.get_cur_video_name()
        cur_video_hash = self.controller.get_cur_video_hash()
        template_name = self.controller.get_template_model_file_name()
        if db:
            db_to_show = np.reshape(db[cur_video_hash][shift]["data"][0, cur_frame_index, :], (7, 2))
        else:
            db_to_show = np.zeros((7, 2))
        sticker_names = ["Left Eye", "Nose Tip", "Right Eye", "CAP1", "CAP2", "CAP3", "CAP4"]

        my_string = "Template Model File: \n{}".format(template_name)
        label = tk.Label(self.data_panel, text=my_string, width=30, bg="white", anchor="center", pady=pad_y)
        label.pack(fill="x")

        my_string = "Video File Name: \n{}".format(cur_video_name)
        label = tk.Label(self.data_panel, text=my_string, width=30, bg="white", anchor="center", pady=pad_y)
        label.pack(fill="x")

        my_string = "Frame: {}".format(self.controller.get_cur_frame_index())
        label = tk.Label(self.data_panel, text=my_string, width=30, bg="white", anchor="center", pady=pad_y)
        label.pack(fill="x")

        for i in range(7):
            my_string = "{}: {},{}".format(sticker_names[i], int(db_to_show[i, 0]), int(db_to_show[i, 1]))
            if i == cur_sticker_index // 2:
                sticker_label = tk.Label(self.data_panel, text=my_string, bg="gray", width=15, anchor="w", pady=pad_y)
            else:
                sticker_label = tk.Label(self.data_panel, text=my_string, bg="white", width=15, anchor="w", pady=pad_y)
            sticker_label.pack(fill="x")
            if db_to_show[i, 0] != 0 and db_to_show[i, 0] != 0:
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


class MainMenu(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.label = tk.Label(self, text="STORM - a fNIRS Calibration Tool", font=("Verdana", 12))
        self.canvas = tk.Canvas(self, height=400, width=400, bg="#263D42")
        img = ImageTk.PhotoImage(master=self, file="resource/render.png")
        self.canvas.img = img  # or else image gets garbage collected
        self.canvas.create_image(0, 0, anchor="nw", image=img, tag="image")

        self.tempalte_view_button = ttk.Button(self, text="View Template Model",
                                               command=lambda: controller.show_panel(ExperimentViewerPage))

        self.finetune_button = ttk.Button(self, text="Fine-tune STORM-Net",
                                          command=lambda: controller.show_panel(FinetunePage))

        self.calibration_button = ttk.Button(self, text="Calibrate",
                                             command=lambda: controller.show_panel(CalibrationPage))
        self.about_button = ttk.Button(self, text="About",
                                             command=lambda: controller.show_panel(AboutPage))

        self.label.grid(row=1, column=1, columnspan=4)
        self.canvas.grid(row=2, column=1, columnspan=4)
        self.tempalte_view_button.grid(row=3, column=1, sticky="w" + "e")
        self.finetune_button.grid(row=3, column=2, sticky="w" + "e")
        self.calibration_button.grid(row=3, column=3, sticky="w" + "e")
        self.about_button.grid(row=3, column=4, sticky="w" + "e")
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(5, weight=1)


class AboutPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.title = tk.Label(self,
                                 text="STORM: Simple and Timely Optode Registration Method for Functional Near-Infrared Spectroscopy (FNIRS).\n"
                                      "Research: Yotam Erel, Sagi Jaffe-Dax, Yaara Yeshurun-Dishon, Amit H. Bermano\n"
                                      "Implementation: Yotam Erel\n"
                                      "This program is free for personal, non-profit or academic use.",
                                 font=("Verdana", 12),
                                 relief="groove",
                                 anchor="w",
                                 justify="left")
        self.button = ttk.Button(self, text="Back",
                                 command=lambda: controller.show_panel(MainMenu))
        self.title.grid(row=1, column=1)
        self.button.grid(row=2, column=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(2, weight=1)


class FinetunePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.title = tk.Label(self,
                                 text="Not implemented yet",
                                 font=("Verdana", 12),
                                 relief="groove",
                                 anchor="w",
                                 justify="left")
        self.button = ttk.Button(self, text="Back",
                                 command=lambda: controller.show_panel(MainMenu))
        self.title.pack()
        self.button.pack()


class ExperimentViewerPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.title = tk.Label(self,
                                 text="Not implemented yet",
                                 font=("Verdana", 12),
                                 relief="groove",
                                 anchor="w",
                                 justify="left")
        self.button = ttk.Button(self, text="Back",
                                 command=lambda: controller.show_panel(MainMenu))
        self.title.pack()
        self.button.pack()


class ProgressBarPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.status_label = tk.Label(self, text="", bg=controller['bg'], pady=10)
        self.prog_bar = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=300, mode="indeterminate")

        self.status_label.pack(side="top")
        self.prog_bar.pack(side="top")

    def show_progress(self, show):
        if show:
            self.prog_bar.start(10)
        else:
            self.prog_bar.stop()
            self.status_label.config(text="")

    def update_status_label(self, label):
        self.status_label.config(text=label)


class GUI(tk.Tk):
    def __init__(self, db, paths, args):
        super().__init__()
        self.db = db
        self.paths = paths
        self.args = args
        self.shift = 0
        self.cur_video_index = 0
        self.cur_frame_index = 0
        self.cur_sticker_index = 0
        self.frames = None
        self.indices = None
        self.queue = queue.Queue()
        self.unet_model = None
        self.storm_model = None
        self.graph = None
        self.template_file_name = None
        self.template_names = None
        self.template_data = None
        self.template_format = None
        self.projected_data = None
        if self.args.mode == "semi-auto" or self.args.mode == "experimental":
            unet_model_name = args.u_net
            unet_model_dir = Path("models")
            unet_model_full_name = Path.joinpath(unet_model_dir, unet_model_name)
            self.unet_model, self.graph = file_io.load_semantic_seg_model(str(unet_model_full_name), self.args.verbosity)
            storm_model_name = args.storm_net
            storm_model_dir = Path("models")
            storm_model_full_name = Path.joinpath(storm_model_dir, storm_model_name)
            self.storm_model, _ = file_io.load_clean_keras_model(storm_model_full_name, self.args.verbosity)
        self.wm_title("STORM - a fNIRS Calibration Tool")
        self.resizable(False, False)
        self.bind("<Escape>", lambda e: self.destroy())
        self.configure(background='white')
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.panels = {}
        for F in (MainMenu, CalibrationPage, ProgressBarPage, AboutPage, FinetunePage, ExperimentViewerPage):
            panel = F(self.container, self)
            self.panels[F] = panel
            panel.grid(row=0, column=0, sticky="nsew")
        self.show_panel(MainMenu)

    def process_queue(self):
        try:
            msg = self.queue.get(False)
            if msg[0] == "calibrate":
                self.projected_data = msg[1]
                self.save_calibration()
            elif msg[0] == "video_to_frames":
                self.frames, self.indices, my_hash = msg[1:]
                if my_hash not in self.db.keys():
                    data = np.zeros((1, 10, 14))
                    my_dict = {"data": data,
                               "label": np.array([0, 0, 0]),
                               "frame_indices": self.indices}
                    self.db[my_hash] = [my_dict]
                self.panels[CalibrationPage].update_canvas()
                self.panels[CalibrationPage].update_labels()
            elif msg[0] == "annotate_frames":
                my_hash, my_dict = msg[1:]
                self.db[my_hash] = [my_dict]
                self.panels[CalibrationPage].update_canvas()
                self.panels[CalibrationPage].update_labels()
            elif msg[0] == "shift_video":
                self.frames, self.indices = msg[1:]
                self.db[self.get_cur_video_hash()][self.shift]["frame_indices"] = self.indices
                if self.args.verbosity:
                    print("new indices:", self.indices)
                self.panels[CalibrationPage].update_canvas()
                self.panels[CalibrationPage].update_labels()
            elif msg[0] == "load_template_model":
                self.template_names, self.template_data, self.template_format, self.template_file_name = msg[1:]
                self.panels[CalibrationPage].update_labels()
            # Show result of the task if needed
            self.panels[ProgressBarPage].show_progress(False)
            self.show_panel(CalibrationPage)
            self.enable_menu()
        except queue.Empty:
            self.after(100, self.process_queue)

    def take_async_action(self, msg):
        msg_dict = {"video_to_frames": "Selecting frames from video...",
                    "annotate_frames": "Predicting landmarks...This might take some time if GPU is not being used.\nUsing GPU: {}".format(predict.is_using_gpu()),
                    "load_template_model": "Loading template model...",
                    "shift_video": "Selecting different frame...",
                    "calibrate": "Calibrating..."}
        self.disbale_menu()
        self.show_panel(ProgressBarPage)
        self.panels[ProgressBarPage].show_progress(True)
        self.panels[ProgressBarPage].update_status_label(msg_dict[msg[0]])
        ThreadedTask(self.queue, msg).start()
        self.after(100, self.process_queue)

    def update_menubar(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load Video", command=self.load_video)
        filemenu.add_command(label="Load Template Model", command=self.load_template_model)
        filemenu.add_command(label="Load Session", command=self.load_session)
        filemenu.add_command(label="Save Session", command=self.save_session)
        filemenu.add_command(label="Save Calibration", command=self.save_calibration)
        filemenu.add_separator()
        filemenu.add_command(label="Back To Main Menu", command=lambda: self.show_panel(MainMenu))
        menubar.add_cascade(label="File", menu=filemenu)
        videomenu = tk.Menu(menubar, tearoff=0)
        videomenu.add_command(label="Next Frame", command=self.next_frame, accelerator="Right")
        videomenu.add_command(label="Previous Frame", command=self.prev_frame, accelerator="Left")
        videomenu.add_command(label="Set (current) Sticker", command=self.set_coords, accelerator="LMB")
        videomenu.add_command(label="Zero (current) Sticker", command=self.zero_coords, accelerator="MMB")
        videomenu.add_command(label="Next Sticker", command=self.next_coords, accelerator="RMB")
        videomenu.add_command(label="Next Video", command=self.next_video)
        videomenu.add_command(label="Previous Video", command=self.prev_video)
        videomenu.add_separator()
        videomenu.add_command(label="Auto Annotate", command=self.auto_annotate)
        videomenu.add_command(label="Calibrate", command=self.calibrate)
        menubar.add_cascade(label="Video", menu=videomenu)
        if self.paths:
            filemenu.entryconfig("Load Session", state="normal")
            filemenu.entryconfig("Save Session", state="normal")
            menubar.entryconfig("Video", state="normal")
            if len(self.paths) == 1:
                videomenu.entryconfig("Next Video", state="disabled")
                videomenu.entryconfig("Previous Video", state="disabled")
            else:
                videomenu.entryconfig("Next Video", state="normal")
                videomenu.entryconfig("Previous Video", state="normal")
            if self.template_file_name:
                videomenu.entryconfig("Calibrate", state="normal")
            else:
                videomenu.entryconfig("Calibrate", state="disabled")
        else:
            filemenu.entryconfig("Load Session", state="disabled")
            filemenu.entryconfig("Save Session", state="disabled")
            menubar.entryconfig("Video", state="disabled")
        if self.projected_data:
            filemenu.entryconfig("Save Calibration", state="normal")
        else:
            filemenu.entryconfig("Save Calibration", state="disabled")
        return menubar

    def show_panel(self, cont):
        panel = self.panels[cont]
        panel.tkraise()
        panel.focus_set()
        if cont == CalibrationPage:
            self.config(menu=self.update_menubar())
        else:
            self.config(menu="")

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

    def prep_vid_to_frames_packet(self):
        path = self.paths[self.cur_video_index]
        return ["video_to_frames",
                path]

    def prep_annotate_frame_packet(self):
        path = self.paths[self.cur_video_index]
        return ["annotate_frames",
                path,
                self.unet_model,
                self.graph,
                self.args]

    def prep_calibrate_packet(self):
        return ["calibrate",
                self.template_names,
                self.template_data,
                self.db[self.get_cur_video_hash()][self.shift]["data"],
                self.storm_model,
                self.graph,
                self.args]

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
        if self.paths:
            parent = self.paths[self.cur_video_index].parent.name
            name = self.paths[self.cur_video_index].name
            my_str = "{}/{}".format(parent, name)
        else:
            my_str = "Not Loaded"
        return my_str

    def get_template_model_file_name(self):
        if self.template_file_name:
            parent = self.template_file_name.parent.name
            name = self.template_file_name.name
            my_str = "{}/{}".format(parent, name)
        else:
            my_str = "Not Loaded"
        return my_str

    def get_cur_video_hash(self):
        if self.paths:
            my_str = utils.md5_from_vid(self.paths[self.cur_video_index])
        else:
            my_str = ""
        return my_str

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
        self.panels[CalibrationPage].update_labels()

    def save_coords(self, event=None, coords=None):
        current_video = self.get_cur_video_hash()
        if coords:
            self.db[current_video][self.shift]["data"][0, self.cur_frame_index,
            self.cur_sticker_index:self.cur_sticker_index + 2] = coords[0], coords[1]
        else:
            self.db[current_video][self.shift]["data"][0, self.cur_frame_index, self.cur_sticker_index:self.cur_sticker_index+2] = event.x, 540-event.y
        self.go_to_next_coord()

    def set_coords(self):
        x = simpledialog.askinteger(title="Set Current Sticker Coordinates (Use mouse for faster annotation)",
                                    prompt="Enter X Value:")
        y = simpledialog.askinteger(title="Set Current Sticker Coordinates (Use mouse for faster annotation)",
                                    prompt="Enter Y Value:")
        if x and y:
            h, w = self.get_frame_size()
            if x < h and y < w:
                self.save_coords(coords=(x, y))
            else:
                if self.args.verbosity:
                    print("Must supply values between 0-539 and 0-959")

    def zero_coords(self, event=None):
        current_video = self.get_cur_video_hash()
        self.db[current_video][self.shift]["data"][0, self.cur_frame_index, self.cur_sticker_index:self.cur_sticker_index+2] = 0, 0
        self.go_to_next_coord()

    def next_coords(self, event=None):
        self.go_to_next_coord()

    def auto_annotate(self):
        if not np.array_equal(self.db[self.get_cur_video_hash()][self.shift]["data"], np.zeros((1, 10, 14))):
            result = messagebox.askquestion("Manual Annotation Detected",
                                            "Are you sure you want to automaticaly annotate the frames? current manual annotations will be lost.",
                                            icon='warning')
            if result != 'yes':
                return
        self.cur_frame_index = 0
        self.cur_sticker_index = 0
        self.take_async_action(self.prep_annotate_frame_packet())

    def calibrate(self):
        if np.array_equal(self.db[self.get_cur_video_hash()][self.shift]["data"], np.zeros((1, 10, 14))):
            result = messagebox.askquestion("No Annotation Detected",
                                            "Are you sure you want to calibrate? frames seem to be not annotated.",
                                            icon='warning')
            if result != 'yes':
                return
        self.take_async_action(self.prep_calibrate_packet())

    def next_video(self):
        if self.cur_video_index < (len(self.paths)-1):
            self.cur_video_index += 1
            self.cur_frame_index = 0
            self.cur_sticker_index = 0
            self.take_async_action(self.prep_vid_to_frames_packet())

    def prev_video(self):
        if self.cur_video_index > 0:
            self.cur_video_index -= 1
            self.cur_frame_index = 0
            self.cur_sticker_index = 0
            self.take_async_action(self.prep_vid_to_frames_packet())

    def shift_video_f(self):
        current_video = self.get_cur_video_hash()
        current_indices = self.db[current_video][self.shift]["frame_indices"]
        if self.args.verbosity:
            print("current indices:", current_indices)
        new_indices = current_indices.copy()
        new_indices[self.get_cur_frame_index()] += 1
        self.take_async_action(["shift_video", self.paths, self.cur_video_index, new_indices])

    def shift_video_b(self):
        current_video = self.get_cur_video_hash()
        current_indices = self.db[current_video][self.shift]["frame_indices"]
        if self.args.verbosity:
            print("current indices:", current_indices)
        new_indices = current_indices.copy()
        new_indices[self.get_cur_frame_index()] -= 1
        self.take_async_action(["shift_video", self.paths, self.cur_video_index, new_indices])

    def next_frame(self, event=None):
        if self.cur_frame_index < 9:
            self.cur_frame_index += 1
            self.cur_sticker_index = 0
            self.panels[CalibrationPage].update_canvas()
            self.panels[CalibrationPage].update_labels()

    def prev_frame(self, event=None):
        if self.cur_frame_index > 0:
            self.cur_frame_index -= 1
            self.cur_sticker_index = 0
            self.panels[CalibrationPage].update_canvas()
            self.panels[CalibrationPage].update_labels()

    def load_video(self):
        filename = filedialog.askopenfilename(initialdir=".", title="Select Video File")
        if not filename:
            return
        file = Path(filename)
        if file.suffix.lower() in [".mp4", ".avi"]:
            self.paths = [file]
            self.take_async_action(self.prep_vid_to_frames_packet())

    def load_template_model(self):
        filename = filedialog.askopenfilename(initialdir=".", title="Select Template Model File")
        if not filename:
            return
        file = Path(filename)
        self.take_async_action(["load_template_model", file])

    def load_session(self):
        if self.paths:
            filename = filedialog.askopenfilename(initialdir=".", title="Select Session File")
            if not filename:
                return
            self.db = file_io.load_from_pickle(filename)
            self.take_async_action(self.prep_vid_to_frames_packet())
        else:
            if self.args.verbosity:
                print("You must first load a video before you can load any saved sessions.")

    def save_session(self):
        f = filedialog.asksaveasfile(initialdir="./data", title="Select Session File", mode='wb')
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        file_io.dump_to_pickle(Path(f.name), self.db)

    def save_calibration(self):
        if self.projected_data:
            f = filedialog.asksaveasfile(initialdir="./data", title="Select Output File", mode='w')
            if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
                return
            file_io.save_results(self.projected_data, Path(f.name), self.args.verbosity)


class ThreadedTask(threading.Thread):
    def __init__(self, queue, msg):
        threading.Thread.__init__(self)
        self.queue = queue
        self.msg = msg

    def run(self):
        if self.msg[0] == "video_to_frames":
            self.handle_video_to_frames()
        elif self.msg[0] == "annotate_frames":
            self.handle_annotate_frames()
        elif self.msg[0] == "shift_video":
            self.handle_shift_video()
        elif self.msg[0] == "load_template_model":
            self.handle_load_template_model()
        elif self.msg[0] == "calibrate":
            self.handle_calibrate()

    def handle_calibrate(self):
        template_names, template_data, data, model, graph, args = self.msg[1:]
        r, s = predict.predict_rigid_transform(data, model, graph, args)
        sensor_locations = geometry.apply_rigid_transform(r, s, template_names, template_data, None, args)
        projected_data = geometry.project_sensors_to_MNI(sensor_locations, args.verbosity)
        self.queue.put(["calibrate", projected_data])

    def handle_load_template_model(self):
        path = self.msg[1]
        template_names, template_data, template_format = geometry.read_template_file(path)
        self.queue.put(["load_template_model", template_names[0], template_data[0], template_format, path])

    def handle_video_to_frames(self):
        path = self.msg[1]
        my_hash = utils.md5_from_vid(path)
        frames, indices = video.video_to_frames(path, vid_hash=my_hash, dump_frames=True)
        self.queue.put(["video_to_frames", frames, indices, my_hash])

    def handle_annotate_frames(self):
        path, preloaded_model, graph, args = self.msg[1:]
        my_hash = utils.md5_from_vid(path)
        frames, indices = video.video_to_frames(path, vid_hash=my_hash, dump_frames=True)
        data = predict.predict_keypoints_locations(frames,
                                                   args,
                                                   my_hash,
                                                   is_puppet=False,
                                                   save_intermed=False,
                                                   preloaded_model=preloaded_model,
                                                   graph=graph)
        my_dict = {"data": data,
                   "label": np.array([0, 0, 0]),
                   "frame_indices": indices}
        self.queue.put(["annotate_frames", my_hash, my_dict])

    def handle_shift_video(self):
        paths, cur_video_index, new_indices = self.msg[1:]
        frames, indices = video.video_to_frames(paths[cur_video_index],
                                                dump_frames=True,
                                                frame_indices=new_indices,
                                                force_reselect=True)
        self.queue.put(["shift_video", frames, indices])


def annotate_videos(args):  # contains GUI mainloop
    if args.mode == "experimental":
        special_db = Path.joinpath(Path("data"), "telaviv_db.pickle")
        new_db = file_io.load_full_db(special_db)
        paths = []
        if args.video:
            if Path.is_file(args.video):
                paths.append(args.video)
            else:
                for file in args.video.glob("**/*.MP4"):
                    paths.append(file)
    else:
        new_db = file_io.load_full_db()
        paths = None
    if args.verbosity:
        print("Launching GUI...")
    app = GUI(new_db, paths, args)
    app.mainloop()
    return app.get_db(), app.paths


# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Automatically annotates FNIRS videos on disk.')
#     parser.add_argument("video_folder", help="The path to the video folder.")
#     parser.add_argument("model_file", help="The base model file path.")
#     parser.add_argument("-a", "--auto_annotate", action='store_true', help="Automatically annotates videos in folder")
#     parser.add_argument("-g", "--gui", action='store_true', help="Shows GUI")
#     # if len(sys.argv) == 1:
#     #     parser.print_help(sys.stderr)
#     #     sys.exit(1)
#     # cmd_line = '/disk1/yotam/capnet/openPos/openPos/openPos49/ /disk1/yotam/capnet/openPos/openPos/openPos46/ '.split()
#     cmd_line = 'E:/University/masters/CapTracking/videos/openPos53/GX011577.MP4 E:/University/masters/CapTracking/videos/openPos53 -g'.split()
#     args = parser.parse_args(cmd_line)  # cmd_line
#     args.video_folder = Path(args.video_folder)
#     args.model_file = Path(args.model_file)
#     if Path.is_dir(args.model_file):
#         args.model_file = args.model_file.glob("*.txt").__next__()
#     return args


# if __name__ == "__main__":
    # blacklist = ["GX011543.MP4", "GX011544.MP4", "GX011547.MP4", "GX011549.MP4",
    #              "GX011537.MP4", "GX011538.MP4"]
    # args = parse_arguments()
    # if args.auto_annotate:
    #     new_db = video.auto_annotate_videos(args.video_folder, args.model_file)
    # if args.gui:
    #     annotate_videos(args.video_folder)
