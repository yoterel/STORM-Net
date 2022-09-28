from pathlib import Path
import video
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, simpledialog, messagebox, scrolledtext
import polyscope as ps
import polyscope.imgui as psim
import predict
from PIL import Image, ImageTk
import numpy as np
import file_io
import os
import time
import queue
import threading
import utils
import geometry
import logging
import webbrowser
import render
import config
import torch
from torch_src import torch_model
import torch_train
import MNI
## globals for gui
prev_selection = None
pc_names = None
pc_path = None
pc_data = None
mni_flag = False
ui_options = ["Original", "MNI", "STORM-Net"]
ui_options_selected = ui_options[0]
prev_ui_options_selected = ui_options_selected
## globals for gui

def post_process_db(db):
    perform_pad = False
    perform_type_conversion = False
    for key in db:
        landmarks_per_frame = db[key][0]["data"].shape[-1] // 2
        diff = config.max_number_of_landmarks_per_frames - landmarks_per_frame
        if diff != 0:
            if perform_pad:
                temp = db[key][0]["data"][:, :, 10:].copy()
                db[key][0]["data"] = np.pad(db[key][0]["data"], ((0, 0), (0, 0), (0, 2*diff)), 'constant')
                db[key][0]["data"][:, :, 10:12] = 0
                db[key][0]["data"][:, :, 12:] = temp
    return db


def annotate_videos(args):  # contains GUI mainloop
    if args.mode == "experimental":
        special_db = Path.joinpath(Path("cache"), "telaviv_db.pickle")
        new_db = file_io.load_full_db(special_db)
        # new_db = post_process_db(new_db)
        paths = []
        if args.video:
            if Path.is_file(args.video):
                paths.append(args.video)
            else:
                for file in args.video.glob("**/*.[mM][pP]4"):
                    paths.append(file)
        if args.headless:
            return new_db
    else:  # mode is gui
        new_db = file_io.load_full_db()
        paths = None
    logging.info("Launching GUI...")
    app = GUI(new_db, paths, args)
    app.mainloop()
    return app.get_db()


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
        self.selected_optode = 0
        self.view_elev = 60
        self.view_azim = 220
        self.view_fiducials_only = False
        self.frames = None
        self.indices = None
        self.queue = queue.Queue(maxsize=10)
        self.periodic_queue = queue.Queue(maxsize=10)
        self.render_thread = None
        self.finetune_thread = None
        self.unet_model = None
        self.storm_model = None
        self.pretrained_stormnet_path = None
        self.unet_graph = None
        self.finetunning = False
        self.renderer_executable = None
        self.synth_output_dir = None
        self.renderer_log_file = None
        self.finetune_log_file = None
        self.template_file_name = None
        self.template_names = None
        self.template_data = None
        self.template_format = None
        self.projected_data = None
        self.cur_active_panel = None
        self.render_thread_alive = False
        # if self.args.mode == "gui" or self.args.mode == "experimental":
        #     unet_model_full_path = Path(args.u_net)
        #     self.unet_model, self.unet_graph = file_io.load_semantic_seg_model(str(unet_model_full_path))
        #     storm_model_full_path = Path(args.storm_net)
        #     self.pretrained_stormnet_path = storm_model_full_path
        #     self.storm_model, self.storm_graph = file_io.load_clean_keras_model(storm_model_full_path)
        self.wm_title("STORM-Net - Simple and Timely Optode Registration Method for fNIRS")
        self.resizable(False, False)
        self.bind("<Escape>", lambda e: self.destroy())
        icon = ImageTk.PhotoImage(file="./resource/icon.png", master=self)
        self.iconphoto(False, icon)
        self.configure(background='white')
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.panels = {}
        ps.set_program_name("STORM-Net: Point Cloud Viewer")
        ps.set_verbosity(0)
        ps.set_give_focus_on_show(True)
        ps.set_print_prefix("STORM-Net")
        ps.set_use_prefs_file(False)
        ps.set_up_dir("z_up")
        ps.set_ground_plane_height_factor(1.0, is_relative=False)
        ps.init()
        ps.look_at((20.0, 20.0, 20.0), (0., 0., 0.))
        ps.set_open_imgui_window_for_user_callback(True)
        # ps.set_build_gui(False)
        ps.set_user_callback(ExperimentViewerPage.selection_callback)
        for F in (MainMenu, CoregistrationPage, ProgressBarPage, AboutPage, FinetunePage, ExperimentViewerPage):
            panel = F(self.container, self)
            self.panels[F] = panel
            panel.grid(row=0, column=0, sticky="nsew")
        self.show_panel(MainMenu)
        if self.paths:
            self.take_async_action(self.prep_vid_to_frames_packet())

    def process_periodic_queue(self):
        """
        periodically consumes from periodic queue as long as some periodic thread is alive.
        note: cleans queue and exists if periodic thread sent a done msg
        :return:
        """
        while not self.periodic_queue.empty():
            msg = self.periodic_queue.get(False)
            if msg[0] == "render_data":
                text = msg[1]
                self.panels[FinetunePage].set_renderer_log_text(text)
            elif msg[0] == "render_done":
                self.render_thread_alive = False
                self.render_thread = None
                self.panels[FinetunePage].update_render_progress_bar(False)
            elif msg[0] == "training_data":
                name, value, iter = msg[1:]
                self.panels[FinetunePage].set_finetune_log_text(name, value, iter)
            elif msg[0] == "training_done":
                self.finetunning = False
                self.finetune_thread = None
                self.panels[FinetunePage].update_finetune_progress_bar(False)
        if self.render_thread_alive:
            self.panels[FinetunePage].update_render_progress_bar(True)
            self.after(100, self.process_periodic_queue)
        elif self.finetunning:
            self.panels[FinetunePage].update_finetune_progress_bar(True)
            self.after(100, self.process_periodic_queue)
        else:
            while not self.periodic_queue.empty():
                _ = self.periodic_queue.get()

    def process_queue(self):
        """
        processes msg queue periodically forever
        :return:
        """
        try:
            msg = self.queue.get(False)
            if msg[0] == "coregister":
                self.projected_data = msg[1]
                self.save_registration()
            elif msg[0] == "video_to_frames":
                self.frames, self.indices, my_hash = msg[1:]
                if my_hash not in self.db.keys():
                    data = np.zeros((1, config.number_of_frames_per_video, 2*config.max_number_of_landmarks_per_frames))
                    my_dict = {"data": data,
                               "label": np.array([0, 0, 0]),
                               "frame_indices": self.indices}
                    self.db[my_hash] = [my_dict]
                self.cur_sticker_index = 0
                self.cur_frame_index = 0
            elif msg[0] == "annotate_frames":
                my_hash, my_dict = msg[1:]
                self.db[my_hash] = [my_dict]
            elif msg[0] == "shift_video":
                self.frames, self.indices = msg[1:]
                self.db[self.get_cur_video_hash()][self.shift]["frame_indices"] = self.indices
                logging.info("new indices: " + str(self.indices))
            elif msg[0] == "load_template_model":
                self.template_names, self.template_data, self.template_format, self.template_file_name = msg[1:]
            elif msg[0] == "load_stormnet":
                self.storm_model = msg[1]
                self.pretrained_stormnet_path = msg[2]
                # self.panels[self.cur_active_panel].update_labels()
            # Show result of the task if needed
            self.panels[ProgressBarPage].show_progress(False)
            self.show_panel(self.cur_active_panel)
        except queue.Empty:
            self.after(100, self.process_queue)

    def take_async_action(self, msg, periodic=False):
        """
        takes an asynchronous action in another thread
        :param msg: the msg to send to the thread
        :param periodic: weather this thread should perform some periodic task or not
        :return:
        """
        msg_dict = {"video_to_frames": "Selecting frames from video...",
                    "annotate_frames": "Predicting landmarks...This might take some time if GPU is not being used.\nUsing GPU: {}".format(predict.is_using_gpu()),
                    "load_template_model": "Loading template model...",
                    "load_stormnet": "Loading Storm-Net model...",
                    "shift_video": "Selecting different frame...",
                    "coregister": "Performing Co-Registeration...",
                    "render": "Creating synthetic data, This might take a while...",
                    "finetune": "Fine-tunning STORM-Net. This might take a while...",
                    "predict": "Predicting..."}

        if periodic:
            if msg[0] == "render_stop":
                if self.render_thread:
                    self.render_thread.join()
            elif msg[0] == "render_start":
                self.render_thread_alive = True
                self.render_thread = ThreadedPeriodicTask(self.periodic_queue, msg)
                self.render_thread.start()
                self.after(100, self.process_periodic_queue)
            elif msg[0] == "finetune_stop":
                if self.finetune_thread:
                    self.finetune_thread.join()
                else:
                    logging.info("Training thread is not alive.")
            elif msg[0] == "finetune_start":
                self.finetunning = True
                self.finetune_thread = ThreadedPeriodicTask(self.periodic_queue, msg)
                self.finetune_thread.start()
                self.after(100, self.process_periodic_queue)
        else:
            self.show_panel(ProgressBarPage)
            self.panels[ProgressBarPage].show_progress(True)
            self.panels[ProgressBarPage].update_status_label(msg_dict[msg[0]])
            ThreadedTask(self.queue, msg).start()
            self.after(100, self.process_queue)

    def update_menubar(self, page):
        """
        updates menubar according to current page
        :param page: the current page
        :return: the menubar
        """
        menubar = tk.Menu(self)
        if page == "calib":
            filemenu = tk.Menu(menubar, tearoff=0)
            filemenu.add_command(label="Load Storm-Net Model", command=self.load_stormnet)
            filemenu.add_command(label="Load Video", command=self.load_video)
            filemenu.add_command(label="Load Template Model", command=self.load_template_model)
            filemenu.add_command(label="Load Session", command=self.load_session)
            filemenu.add_command(label="Save Session", command=self.save_session)
            filemenu.add_command(label="Save Registration", command=self.save_registration)
            filemenu.add_separator()
            filemenu.add_command(label="Back To Main Menu", command=lambda: self.show_panel(MainMenu))
            menubar.add_cascade(label="File", menu=filemenu)
            videomenu = tk.Menu(menubar, tearoff=0)
            videomenu.add_command(label="Next Frame", command=self.next_frame, accelerator="Right")
            videomenu.add_command(label="Previous Frame", command=self.prev_frame, accelerator="Left")
            videomenu.add_command(label="Shift Current Frame Forward", command=self.shift_cur_frame_forward, accelerator="Shift + Right")
            videomenu.add_command(label="Shift Current Frame Backward", command=self.shift_cur_frame_backward, accelerator="Shift + Left")
            videomenu.add_command(label="Set (current) Sticker", command=self.set_coords, accelerator="LMB")
            videomenu.add_command(label="Zero (current) Sticker", command=self.zero_coords, accelerator="MMB")
            videomenu.add_command(label="Next Sticker", command=self.next_coords, accelerator="RMB")
            videomenu.add_command(label="Next Video", command=self.next_video)
            videomenu.add_command(label="Previous Video", command=self.prev_video)
            videomenu.add_separator()
            videomenu.add_command(label="Auto Annotate", command=self.auto_annotate)
            videomenu.add_command(label="Co-Register", command=self.coregister)
            menubar.add_cascade(label="Video", menu=videomenu)
            if self.paths:
                # a video is loaded
                filemenu.entryconfig("Load Session", state="normal")
                filemenu.entryconfig("Save Session", state="normal")
                menubar.entryconfig("Video", state="normal")
                if len(self.paths) == 1:
                    # a single video was loaded
                    videomenu.entryconfig("Next Video", state="disabled")
                    videomenu.entryconfig("Previous Video", state="disabled")
                else:
                    videomenu.entryconfig("Next Video", state="normal")
                    videomenu.entryconfig("Previous Video", state="normal")
                if self.template_file_name and self.storm_model:
                    # a template file was loaded
                    videomenu.entryconfig("Co-Register", state="normal")
                else:
                    videomenu.entryconfig("Co-Register", state="disabled")
            else:
                filemenu.entryconfig("Load Session", state="disabled")
                filemenu.entryconfig("Save Session", state="disabled")
                menubar.entryconfig("Video", state="disabled")
            if self.projected_data:
                # coregistration was performed
                filemenu.entryconfig("Save Registration", state="normal")
            else:
                filemenu.entryconfig("Save Registration", state="disabled")
        elif page == "exp":
            filemenu = tk.Menu(menubar, tearoff=0)
            filemenu.add_separator()
            filemenu.add_command(label="Back To Main Menu", command=lambda: self.show_panel(MainMenu))
            menubar.add_cascade(label="File", menu=filemenu)
            optionsmenu = tk.Menu(menubar, tearoff=0)
        elif page == "finetune":
            filemenu = tk.Menu(menubar, tearoff=0)
            filemenu.add_command(label="Load Template Model", command=self.load_template_model)
            filemenu.add_separator()
            filemenu.add_command(label="Back To Main Menu", command=lambda: self.show_panel(MainMenu))
            menubar.add_cascade(label="File", menu=filemenu)
            optionsmenu = tk.Menu(menubar, tearoff=0)
            optionsmenu.add_command(label="Render Synthetic Data", command=self.render)
            optionsmenu.add_command(label="Train STORM-net", command=self.finetune)
            menubar.add_cascade(label="Options", menu=optionsmenu)
            if self.template_file_name:
                menubar.entryconfig("Options", state="normal")
            else:
                menubar.entryconfig("Options", state="disabled")
            if self.renderer_executable and self.synth_output_dir and self.template_file_name:
                optionsmenu.entryconfig("Render Synthetic Data", state="normal")
                optionsmenu.entryconfig("Train STORM-net", state="normal")
            else:
                optionsmenu.entryconfig("Render Synthetic Data", state="disabled")
                optionsmenu.entryconfig("Train STORM-net", state="disabled")
        else:
            menubar = ""
        return menubar

    def show_panel(self, pan):
        """
        shows a panel and possibly updates its display
        :param pan: the panel to show
        :return:
        """
        panel = self.panels[pan]
        panel.tkraise()
        panel.focus_set()
        if pan != ProgressBarPage:
            self.cur_active_panel = pan
        if pan == CoregistrationPage:
            self.config(menu=self.update_menubar("calib"))
            self.panels[pan].update_canvas()
            self.panels[pan].update_labels()
        elif pan == ExperimentViewerPage:
            self.config(menu=self.update_menubar("exp"))
            self.panels[pan].update_labels()
        elif pan == FinetunePage:
            self.config(menu=self.update_menubar("finetune"))
            self.panels[pan].update_labels()
        else:
            self.config(menu=self.update_menubar(""))

    def select_from_filesystem(self, isdir, exists, initial_dir, title):
        """
        selects a filesystem object and returns its path
        :param isdir: is object to select a dir? (else assumes it is a file)
        :param exists: does the filesystem object exist? ignored if isdir is true
        :param initial_dir: the initial popup dialog dir
        :param title: the title of the popup dialog
        :return: the path to the object or None if failure occurs
        """
        if isdir:
            folder = filedialog.askdirectory(initialdir=initial_dir, title=title)
            if not folder:
                return None
            else:
                return Path(folder)
        else:
            if exists:
                file = filedialog.askopenfilename(initialdir=initial_dir, title=title)
                if not file:
                    return None
                else:
                    return Path(file)
            else:
                file = filedialog.asksaveasfile(initialdir=initial_dir, title=title)
                if not file:
                    return None
                else:
                    # todo: okay to forget handle?
                    return Path(file.name)

    ### CoregistrationPage ###
    def prep_vid_to_frames_packet(self, indices=None):
        path = self.paths[self.cur_video_index]
        return ["video_to_frames",
                path,
                indices]

    def prep_annotate_frame_packet(self):
        path = self.paths[self.cur_video_index]
        return ["annotate_frames",
                path,
                self.unet_model,
                self.unet_graph,
                self.args]

    def prep_coregister_packet(self):
        return ["coregister",
                self.template_names,
                self.template_data,
                self.db[self.get_cur_video_hash()][self.shift]["data"].copy(),
                self.storm_model,
                self.args]

    def prep_predict_packet(self):
        cur_path = self.paths[self.cur_video_index]
        assert "session1" in str(cur_path)
        next_path = self.paths[self.cur_video_index + 1]
        cur_data = self.db[self.get_cur_video_hash()][self.shift]["data"].copy()
        next_data = self.db[utils.md5_from_vid(next_path)][self.shift]["data"].copy()
        video_names = [cur_path.parent.name + "_" + cur_path.name, next_path.parent.name + "_" + next_path.name]
        return ["predict",
                np.concatenate((cur_data, next_data)),
                video_names,
                self.args]


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
        if self.paths:
            # parent = self.paths[self.cur_video_index].parent.name
            name = self.paths[self.cur_video_index].name
            my_str = "{}".format(name)
        else:
            my_str = "Not Loaded"
        return my_str

    def get_template_info(self):
        return self.template_names, self.template_data, self.template_format

    def get_template_model_file_name(self):
        if self.template_file_name:
            # parent = self.template_file_name.parent.name
            name = self.template_file_name.name
            my_str = "{}".format(name)
        else:
            my_str = "Not Set"
        return my_str

    def get_renderer_file_name(self):
        if self.renderer_executable:
            parent = self.renderer_executable.parent.name
            name = self.renderer_executable.name
            my_str = "{}/{}".format(parent, name)
        else:
            my_str = "Not Set"
        return my_str

    def get_synth_output_folder_name(self):
        if self.synth_output_dir:
            parent = self.synth_output_dir.parent.name
            name = self.synth_output_dir.name
            my_str = "{}/{}".format(parent, name)
        else:
            my_str = "Not Set"
        return my_str

    def get_renderer_log_file_name(self):
        if self.renderer_log_file:
            parent = self.renderer_log_file.parent.name
            name = self.renderer_log_file.name
            my_str = "{}/{}".format(parent, name)
        else:
            my_str = "Not Set"
        return my_str

    def get_finetune_log_file_name(self):
        if self.finetune_log_file:
            parent = self.finetune_log_file.parent.name
            name = self.finetune_log_file.name
            my_str = "{}/{}".format(parent, name)
        else:
            my_str = "Not Set"
        return my_str

    def get_pretrained_stormnet_path(self):
        if self.pretrained_stormnet_path:
            # parent = self.pretrained_stormnet_path.parent.name
            name = self.pretrained_stormnet_path.name
            my_str = "{}".format(name)
        else:
            my_str = "Not Set"
        return my_str

    def get_cur_video_hash(self):
        if self.paths:
            my_str = utils.md5_from_vid(self.paths[self.cur_video_index])
        else:
            my_str = ""
        return my_str

    def go_to_next_coord(self):
        """
        moves to next sticker and updates display of coregistration page
        :return:
        """
        if self.cur_sticker_index >= 2*(config.max_number_of_landmarks_per_frames - 1):
            self.cur_sticker_index = 0
        else:
            self.cur_sticker_index += 2
        self.panels[CoregistrationPage].update_labels()

    def save_coords(self, event=None, coords=None):
        if self.frames:
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
                logging.info("Must supply values between 0-539 and 0-959")

    def zero_coords(self, event=None):
        """
        sets location to x=0, y=0 for current sticker in current frame of current video.
        Note: 0,0 is considered as "ignore" for STORM-Net
        :param event:
        :return:
        """
        if self.frames:
            current_video = self.get_cur_video_hash()
            self.db[current_video][self.shift]["data"][0, self.cur_frame_index, self.cur_sticker_index:self.cur_sticker_index+2] = 0, 0
            self.go_to_next_coord()

    def next_coords(self, event=None):
        if self.frames:
            self.go_to_next_coord()

    def auto_annotate(self):
        """
        auto annotates a video (in a different thread)
        :return:
        """
        if np.any(self.db[self.get_cur_video_hash()][self.shift]["data"]):
            result = messagebox.askquestion("Manual Annotation Detected",
                                            "Are you sure you want to automaticaly annotate the frames? current manual annotations will be lost.",
                                            icon='warning')
            if result != 'yes':
                return
        self.take_async_action(self.prep_annotate_frame_packet())

    def coregister(self):
        """
        calculates registration from a video (in a different thread)
        :return:
        """
        if not np.any(self.db[self.get_cur_video_hash()][self.shift]["data"]):
            result = messagebox.askquestion("No Annotation Detected",
                                            "Are you sure you want to co-register? frames seem to be not annotated.",
                                            icon='warning')
            if result != 'yes':
                return
        result = messagebox.askyesno("Transform to MNI?",
                                     "Would you like the registered data to be transformed to statistical MNI coordinates?")
        if result:
            self.args.mni = True
        else:
            self.args.mni = False
        self.take_async_action(self.prep_coregister_packet())

    def predict_x(self):
        self.take_async_action(self.prep_predict_packet())

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

    def shift_cur_frame_forward(self, event=None):
        current_video = self.get_cur_video_hash()
        current_indices = self.db[current_video][self.shift]["frame_indices"]
        logging.info("current indices: " + str(current_indices))
        new_indices = current_indices.copy()
        new_indices[self.get_cur_frame_index()] += 1
        self.take_async_action(["shift_video", self.paths[self.cur_video_index], new_indices])

    def shift_cur_frame_backward(self, event=None):
        current_video = self.get_cur_video_hash()
        current_indices = self.db[current_video][self.shift]["frame_indices"]
        logging.info("current indices: " + str(current_indices))
        new_indices = current_indices.copy()
        new_indices[self.get_cur_frame_index()] -= 1
        self.take_async_action(["shift_video", self.paths[self.cur_video_index], new_indices])

    def next_frame(self, event=None):
        if self.cur_frame_index < config.number_of_frames_per_video - 1:
            self.cur_frame_index += 1
            self.cur_sticker_index = 0
            self.panels[CoregistrationPage].update_canvas()
            self.panels[CoregistrationPage].update_labels()

    def prev_frame(self, event=None):
        if self.cur_frame_index > 0:
            self.cur_frame_index -= 1
            self.cur_sticker_index = 0
            self.panels[CoregistrationPage].update_canvas()
            self.panels[CoregistrationPage].update_labels()

    def load_video(self):
        obj = self.select_from_filesystem(False, True, ".", "Select Video File")
        if obj:
            if obj.suffix.lower() in [".mp4", ".avi"]:
                self.paths = [obj]
                self.take_async_action(self.prep_vid_to_frames_packet())

    def load_stormnet(self):
        obj = self.select_from_filesystem(False, True, ".", "Select Storm-Net Model File")
        if obj:
            self.take_async_action(["load_stormnet", obj, self.args.device])

    def load_template_model(self):
        obj = self.select_from_filesystem(False, True, ".", "Select Template Model File")
        if obj:
            self.take_async_action(["load_template_model", obj])

    def load_session(self):
        if self.paths:
            obj = self.select_from_filesystem(False, True, ".", "Select Session File")
            if obj:
                self.db = file_io.load_from_pickle(obj)
                self.take_async_action(self.prep_vid_to_frames_packet())
        else:
            logging.info("You must first load a video before you can load any saved sessions.")

    def save_session(self):
        f = filedialog.asksaveasfile(initialdir=".", title="Select Session File", mode='wb')
        if f is None:
            return
        file_io.dump_to_pickle(Path(f.name), self.db)

    def save_registration(self):
        if self.projected_data:
            f = filedialog.asksaveasfile(initialdir=".", title="Select Output File", mode='w')
            if f is None:
                return
            file_io.save_results(self.projected_data, Path(f.name))

    def set_default_render(self):
        """
        sets default rendering settings in Finetune page
        :return:
        """
        self.take_async_action(["load_template_model", Path("./../example_models/example_model.txt")])
        self.renderer_log_file = Path("./cache/render_log.txt")
        self.synth_output_dir = Path("./cache/synth_data")
        self.synth_output_dir.mkdir(parents=True, exist_ok=True)
        self.renderer_executable = Path("./../DataSynth/windows_build/DataSynth.exe")
        self.panels[FinetunePage].render_set_defaults()
        self.show_panel(FinetunePage)

    def set_default_finetune(self):
        """
        sets default finetuning settings in Finetune page
        :return:
        """
        self.pretrained_stormnet_path = "my_new_model"
        self.finetune_log_file = Path("./cache/training_log.txt")
        self.synth_output_dir = Path("./cache/synth_data")
        self.synth_output_dir.mkdir(parents=True, exist_ok=True)
        self.panels[self.cur_active_panel].finetune_set_defaults()
        self.show_panel(self.cur_active_panel)

    def get_gpu_id(self):
        return self.args.gpu_id

    def load_renderer_executable(self):
        obj = self.select_from_filesystem(False, True, "./..", "Select Renderer Executable File")
        if obj:
            self.renderer_executable = obj
            self.show_panel(self.cur_active_panel)

    def load_synth_output_folder(self):
        obj = self.select_from_filesystem(True, False, "./..", "Select Synthetic Data Output Directory")
        if obj:
            self.synth_output_dir = obj
            self.show_panel(self.cur_active_panel)

    def load_renderer_log_file(self):
        obj = self.select_from_filesystem(False, False, "./..", "Select Renderer Log File Location")
        if obj:
            self.renderer_log_file = obj
            self.show_panel(self.cur_active_panel)

    def load_finetune_log_file(self):
        obj = self.select_from_filesystem(False, False, "./..", "Select Finetune Procedure Log File Location")
        if obj:
            self.finetune_log_file = obj
            self.show_panel(self.cur_active_panel)

    def load_pretrained_model(self):
        obj = self.select_from_filesystem(False, True, "./..", "Select Pretrained Model Location")
        if obj:
            self.pretrained_stormnet_path = Path(obj)
            self.show_panel(self.cur_active_panel)

    def finetune_kill_thread(self):
        self.take_async_action(["finetune_stop"], periodic=True)


    def render_monitor_progress(self):
        """
        callback for the event: user checked the "monitor progress" for rendering.
        :return:
        """
        if not self.panels[FinetunePage].render_monitor_progress.get():
            self.take_async_action(["render_stop"], periodic=True)
            # self.panels[FinetunePage].renderer_log_text.configure(state='normal')
            # self.panels[FinetunePage].renderer_log_text.delete(1.0, tk.END)
            # self.panels[FinetunePage].renderer_log_text.configure(state='disabled')
        else:
            if not self.render_thread_alive and self.is_renderer_active() and self.renderer_log_file.is_file():
                self.take_async_action(["render_start", self.renderer_log_file.absolute()], periodic=True)

    def is_renderer_active(self):
        if self.renderer_executable:
            return file_io.is_process_active(self.renderer_executable.name)
        else:
            return False

    def render(self):
        """
        performs rendering by launching an external executable
        :return:
        """
        if self.template_names is None or self.template_data is None:
            logging.info("Missing template model file.")
            return
        elif self.renderer_executable is None:
            logging.info("Missing renderer executable.")
            return
        elif self.synth_output_dir is None:
            logging.info("Missing synthetic data output folder")
            return
        elif self.renderer_log_file is None:
            logging.info("Missing log file path.")
            return
        elif self.is_renderer_active():
            logging.info("Renderer executable already running.")
            return
        elif self.render_thread_alive:
            logging.info("Already monitoring a log file.")
            return
        elif self.finetunning:
            logging.info("Cannot render while fine-tunning is already in progress.")
            return
        else:
            iterations = self.panels[self.cur_active_panel].iterations_number.get()
            try:
                iterations = int(iterations)
                if iterations < 1:
                    raise ValueError
            except ValueError:
                logging.info("Number of iterations invalid.")
                return

        is_data_folder_empty = not any(self.synth_output_dir.iterdir())
        if not is_data_folder_empty:
            result = messagebox.askquestion("Synthetic Data Folder Not Empty",
                                            "The synthetic data folder you provided as output is not empty, this action will delete any content in the folder. Proceed?",
                                            icon='warning')
            if result != 'yes':
                return
            else:
                file_io.delete_content_of_folder(self.synth_output_dir)
        log_file_exist = self.renderer_log_file.is_file()
        if log_file_exist:
            result = messagebox.askquestion("Renderer Log File Exists",
                                            "The log file you provided for the renderer exists, this action will overwrite this file. Proceed?",
                                            icon='warning')
            if result != 'yes':
                return
            else:
                _ = open(self.renderer_log_file.absolute(), "w+")
        status, _ = render.render(self.template_names,
                                  self.template_data,
                                  self.synth_output_dir,
                                  self.renderer_executable,
                                  self.renderer_log_file,
                                  iterations,
                                  False,
                                  False)
        if status:
            if self.panels[self.cur_active_panel].render_monitor_progress.get():
                self.take_async_action(["render_start", self.renderer_log_file.absolute()], periodic=True)

    def finetune(self):
        model_name = self.panels[self.cur_active_panel].model_name.get()
        if model_name == "":
            logging.info("Missing new model name.")
            return
        elif self.synth_output_dir is None:
            logging.info("Missing synthetic data output folder")
            return
        elif self.finetune_log_file is None:
            logging.info("Missing log file path.")
            return
        elif self.is_renderer_active():
            logging.info("Cannot fine-tune while renderer executable is running.")
            return
        elif self.finetunning:
            logging.info("Training is already in progress.")
            return
        else:
            self.take_async_action(self.prep_fintune_packet(model_name), periodic=True)

    def prep_fintune_packet(self, model_name):
        return ["finetune_start", model_name, self.synth_output_dir, self.args.device]

    def toggle_optodes(self, event=None):
        self.selected_optode = 0
        self.view_fiducials_only = not self.view_fiducials_only
        self.panels[ExperimentViewerPage].update_labels()

    def next_optode(self, event=None):
        if self.view_fiducials_only:
            spiral_index = self.template_names.index(0)
            if self.selected_optode < spiral_index - 2:
                self.selected_optode += 1
                self.panels[ExperimentViewerPage].update_labels()
            else:
                return
        else:
            if self.selected_optode < len(self.template_data) - 1:
                self.selected_optode += 1
                self.panels[ExperimentViewerPage].update_labels()
            else:
                return

    def prev_optode(self, event=None):
        if self.selected_optode > 0:
            self.selected_optode -= 1
            self.panels[ExperimentViewerPage].update_labels()
        else:
            return

    def get_view_elev(self):
        return self.view_elev

    def get_view_azim(self):
        return self.view_azim

    def increase_azim(self, event=None):
        self.view_azim += 10
        self.panels[ExperimentViewerPage].update_labels()

    def decrease_azim(self, event=None):
        self.view_azim -= 10
        self.panels[ExperimentViewerPage].update_labels()

    def increase_elev(self, event=None):
        self.view_elev += 10
        self.panels[ExperimentViewerPage].update_labels()

    def decrease_elev(self, event=None):
        self.view_elev -= 10
        self.panels[ExperimentViewerPage].update_labels()

    def fiducials_only(self):
        return self.view_fiducials_only


class ThreadedPeriodicTask(threading.Thread):
    def __init__(self, queue, msg):
        threading.Thread.__init__(self)
        self.queue = queue
        self.msg = msg
        self.stoprequest = threading.Event()

    def join(self, timeout=None):
        self.stoprequest.set()

    def run(self):
        if self.msg[0] == "finetune_start":
            self.handle_finetune()
        elif self.msg[0] == "render_start":
            self.handle_render()

    def handle_finetune(self):
        model_name, synth_output_dir, device = self.msg[1:]

        class Options:
            def __init__(self):
                self.experiment_name = model_name
                self.data_path = synth_output_dir
                self.architecture = "2dconv"
                self.force_load_raw_data = False
                self.loss = "l2"
                self.scale_faces = None
                self.dont_use_gmm = False
                self.device = device
                self.continue_train = False
                self.batch_size = 16
                self.number_of_epochs = 50
                self.lr = 1e-4
                self.beta1 = 0.9
                # self.template = Path("../example_models/example_model.txt")
                self.network_input_size = 10
                self.num_threads = 0
                self.log = False
                self.tensorboard = None
                self.create_new_checkpoints_per_epoch = False
                self.verbosity = "info"
                self.is_train = True
                self.network_output_size = 3
                self.root = Path("models", self.experiment_name)
        opt = Options()
        try:
            torch_train.train_loop(opt, [self.stoprequest, self.queue])
            # train.train(model_name, synth_output_dir, pretrained_stormnet_path, None, 0, Path("models"), self.queue,
            #             self.stoprequest)
        except IndexError:
            logging.warning("Training STORM-Net failed. Maybe the synthetic data is incorrect / corrupted ?")
            self.queue.put(["training_done"])

    def handle_render(self):
        path = self.msg[1]
        while not path.is_file() and not self.stoprequest.isSet():
            time.sleep(0.1)
        if path.is_file():
            logfile = open(str(path), "r")
            '''generator function that yields new lines in a file
            '''
            # seek the end of the file
            logfile.seek(0, os.SEEK_END)

            # start infinite loop
            while not self.stoprequest.isSet():
                # read last line of file
                line = logfile.readline()  # sleep if file hasn't been updated
                if not line:
                    time.sleep(0.1)
                    continue
                if "Done" in line:
                    self.join()
                else:
                    if not self.queue.full():
                        self.queue.put(["render_data", line])
        self.queue.put(["render_done"])


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
        elif self.msg[0] == "load_stormnet":
            self.handle_load_stormnet()
        elif self.msg[0] == "coregister":
            self.handle_coregister()
        elif self.msg[0] == "predict":
            self.handle_predict()

    def handle_predict(self):
        data, video_names, args = self.msg[1:]
        subject_name, _ = video_names[0].split("_")
        subject_name = subject_name + ".txt"
        r, s = predict.predict_rigid_transform(data, None, args)
        self.queue.put(["predict"])

    def handle_coregister(self):
        template_names, template_data, data, model, args = self.msg[1:]
        r, s = predict.predict_rigid_transform(data, model, args)
        sensor_locations = geometry.apply_rigid_transform(r, s, template_names, template_data, None, args)
        if args.mni:
            projected_data = geometry.project_sensors_to_MNI(sensor_locations, transform_anchors=True)
        else:
            projected_data = sensor_locations
        self.queue.put(["coregister", projected_data[0]])

    def handle_load_template_model(self):
        path = self.msg[1]
        try:
            template_names, template_data, template_format, _ = geometry.read_template_file(path)
        except (TypeError, IndexError) as e:
            template_names, template_data, template_format, path = [None], [None], None, None
        self.queue.put(["load_template_model", template_names[0], template_data[0], template_format, path])

    def handle_load_stormnet(self):
        path = Path(self.msg[1])
        device = self.msg[2]
        if path.suffix.lower() in [".pth", ".h5"]:
            model_full_path = path
            class MyOptions:
                def __init__(self, device):
                    self.network_input_size = 10
                    self.architecture = "2dconv"
                    self.loss = "l2"
                    self.device = device
                    self.scale_faces = None
                    self.network_output_size = 3
            opt = MyOptions(device=device)
            network = torch_model.MyNetwork(opt)
            state_dict = torch.load(model_full_path, map_location=device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            network.load_state_dict(state_dict)
            network.to(opt.device)
            self.queue.put(["load_stormnet", network, path])

    def handle_video_to_frames(self):
        path, indices = self.msg[1:]
        my_hash = utils.md5_from_vid(path)
        frames, indices = video.video_to_frames(path, vid_hash=my_hash, dump_frames=True, frame_indices=indices)
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
        path, new_indices = self.msg[1:]
        my_hash = utils.md5_from_vid(path)
        frames, indices = video.video_to_frames(path,
                                                vid_hash=my_hash,
                                                dump_frames=True,
                                                frame_indices=new_indices,
                                                force_reselect=True)
        self.queue.put(["shift_video", frames, indices])


class CoregistrationPage(tk.Frame):
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
        self.bind("<Shift-Left>", self.controller.shift_cur_frame_backward)
        self.bind("<Shift-Right>", self.controller.shift_cur_frame_forward)
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
        storm_net_name = self.controller.get_pretrained_stormnet_path()
        sticker_names = config.virtual_sticker_names
        num_of_stickers = len(sticker_names)
        db_to_show = np.zeros((num_of_stickers, 2))
        if db:
            if cur_video_hash in db.keys():
                db_to_show = np.reshape(db[cur_video_hash][shift]["data"][0, cur_frame_index, :2*num_of_stickers], (num_of_stickers, 2))

        my_string = "Template Model File: \n{}".format(template_name)
        label = tk.Label(self.data_panel, text=my_string, width=30, bg="white", anchor="center", pady=pad_y)
        label.pack(fill="x")

        my_string = "STORM-Net Model: \n{}".format(storm_net_name)
        label = tk.Label(self.data_panel, text=my_string, width=30, bg="white", anchor="center", pady=pad_y)
        label.pack(fill="x")

        my_string = "Video File Name: \n{}".format(cur_video_name)
        label = tk.Label(self.data_panel, text=my_string, width=30, bg="white", anchor="center", pady=pad_y)
        label.pack(fill="x")

        my_string = "Frame: {}".format(self.controller.get_cur_frame_index())
        label = tk.Label(self.data_panel, text=my_string, width=30, bg="white", anchor="center", pady=pad_y)
        label.pack(fill="x")

        for i in range(len(sticker_names)):
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

        # button = ttk.Button(self.data_panel, text="Predict", command=lambda: self.controller.predict_x())
        # button.pack(fill="x")

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
        self.label = tk.Label(self, text="STORM-Net Registration Toolbox", font=("Verdana", 12))
        self.canvas = tk.Canvas(self, height=400, width=400, bg="#263D42")
        img = ImageTk.PhotoImage(master=self, file="resource/render.png")
        self.canvas.img = img  # or else image gets garbage collected
        self.canvas.create_image(0, 0, anchor="nw", image=img, tag="image")

        self.tempalte_view_button = ttk.Button(self, text="View Template Model",
                                               command=lambda: controller.show_panel(ExperimentViewerPage))

        self.finetune_button = ttk.Button(self, text="Offline Step",
                                          command=lambda: controller.show_panel(FinetunePage))

        self.coregistration_button = ttk.Button(self, text="Online Step",
                                             command=lambda: controller.show_panel(CoregistrationPage))
        self.about_button = ttk.Button(self, text="About",
                                             command=lambda: controller.show_panel(AboutPage))

        self.label.grid(row=1, column=1, columnspan=4)
        self.canvas.grid(row=2, column=1, columnspan=4)
        self.tempalte_view_button.grid(row=3, column=1, sticky="w" + "e")
        self.finetune_button.grid(row=3, column=2, sticky="w" + "e")
        self.coregistration_button.grid(row=3, column=3, sticky="w" + "e")
        self.about_button.grid(row=3, column=4, sticky="w" + "e")
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(5, weight=1)


class AboutPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.title = tk.Label(self,
                                 text="STORM-Net: Simple and Timely Optode Registration Method for Functional Near-Infrared Spectroscopy (FNIRS).\n"
                                      "Research: Yotam Erel, Sagi Jaffe-Dax, Yaara Yeshurun-Dishon, Amit H. Bermano\n"
                                      "Implementation: Yotam Erel\n"
                                      "This program is free for personal, non-profit or academic use.\n"
                                      "All Rights Reserved.",
                                 font=("Verdana", 12),
                                 relief="groove",
                                 anchor="w",
                                 justify="left")
        self.subtitle = tk.Label(self, text="https://github.com/yoterel/STORM-Net", fg="blue", cursor="hand2")
        self.button = ttk.Button(self, text="Back",
                                 command=lambda: controller.show_panel(MainMenu))
        self.title.grid(row=1, column=1)
        self.subtitle.grid(row=2, column=1)
        self.subtitle.bind("<Button-1>", lambda e: webbrowser.open_new("https://github.com/yoterel/STORM-Net"))
        self.button.grid(row=3, column=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(2, weight=1)


class ExperimentViewerPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

    def update_labels(self):
        ps.show()
        self.controller.show_panel(MainMenu)

    def selection_callback():
        global prev_selection, pc_names, pc_data, pc_path, ui_options_selected, prev_ui_options_selected, mni_flag
        if(psim.Button("Load Template Model")):
        # This code is executed when the button is pressed
            ps.remove_all_structures()
            path = file_io.select_from_filesystem(False, True, ".", "Select Template Model File")
            if path:
                names, data, _, _ = file_io.read_template_file(path)
                pc_names = names[0]
                pc_data = data[0]
                pc_path = path
                fill_structures(pc_names, pc_data, path.name, ui_options_selected)
        psim.PushItemWidth(200)
        changed = psim.BeginCombo("Visualization Mode", ui_options_selected)
        if changed:
            for val in ui_options:
                _, selected = psim.Selectable(val, ui_options_selected==val)
                if selected:
                    ui_options_selected = val
                    if prev_ui_options_selected != ui_options_selected:
                        prev_ui_options_selected = ui_options_selected
                        if pc_names is not None and pc_data is not None and pc_path is not None:
                            ps.remove_all_structures()
                            fill_structures(pc_names, pc_data, pc_path, ui_options_selected, force_transform=mni_flag)
            psim.EndCombo()
        psim.PopItemWidth() 
        changed, mni_flag = psim.Checkbox("Transform to MNI?", mni_flag) 
        if(changed):
            if pc_names is not None and pc_data is not None and pc_path is not None:
                ps.remove_all_structures()
                fill_structures(pc_names, pc_data, pc_path, ui_options_selected, force_transform=mni_flag)
        if ps.have_selection():
            cur_selection = ps.get_selection()
            psim.TextUnformatted("Selection ID: {}".format(pc_names[cur_selection[1]]))
            if cur_selection != prev_selection:
                pc_name, v_index = cur_selection
                prev_selection = cur_selection
                pc = ps.get_point_cloud(pc_name)
                if 0 <= v_index <= pc.n_points():
                    anchor_mask = np.isin(np.array(pc_names), np.array(config.all_possible_anchor_names))
                    sticker_mask = np.isin(np.array(pc_names), np.array(config.physical_sticker_names))
                    colors = np.ones((pc.n_points(), 3))
                    colors[anchor_mask] = 0
                    colors[sticker_mask] = np.array([0, 1, 0])
                    colors[v_index] = np.array([0., 0., 1.])
                    pc.add_color_quantity("colors",
                                           colors,
                                           enabled=True)


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


class FinetunePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.render_monitor_progress = tk.BooleanVar(self)
        self.render_frame = tk.Frame(self, borderwidth=1, relief=tk.GROOVE)
        self.finetune_frame = tk.Frame(self, borderwidth=1, relief=tk.GROOVE)
        self.render_frame_title = tk.Label(self, text="Render Synthetic Data", pady=10)
        self.finetune_frame_title = tk.Label(self, text="Train STORM-Net", pady=10)
        self.template_name_static = tk.Label(self.render_frame, text="", pady=10)
        self.template_name = tk.Label(self.render_frame, text="", bg=controller['bg'], pady=10)
        self.template_name_button = tk.Button(self.render_frame, text="...",
                                              command=self.controller.load_template_model)
        self.renderer_name_static = tk.Label(self.render_frame, text="", pady=10)
        self.renderer_name = tk.Label(self.render_frame, text="", bg=controller['bg'], pady=10)
        self.renderer_name_button = tk.Button(self.render_frame, text="...",
                                              command=self.controller.load_renderer_executable)
        self.output_folder_static = tk.Label(self.render_frame, text="", pady=10)
        self.output_folder = tk.Label(self.render_frame, text="", bg=controller['bg'], pady=10)
        self.output_folder_button = tk.Button(self.render_frame, text="...",
                                              command=self.controller.load_synth_output_folder)
        self.output_log_label = tk.Label(self.render_frame, text= "", pady=10)
        self.output_log_button = tk.Button(self.render_frame, text="...",
                                           command=self.controller.load_renderer_log_file)
        self.output_log_name = tk.Label(self.render_frame, text="", bg=controller['bg'], pady=10)
        self.iterations_number_label = tk.Label(self.render_frame, text="Number Of Iterations", pady=10)
        self.iterations_number = tk.Entry(self.render_frame, text="", bg=controller['bg'])
        self.render_monitor_progress_checkbox = tk.Checkbutton(self.render_frame, text="Monitor Progress?",
                                                               variable=self.render_monitor_progress,
                                                               command=self.controller.render_monitor_progress)
        self.render_default_button = tk.Button(self.render_frame, text="Default Settings",
                                               command=self.controller.set_default_render)
        self.render_button = tk.Button(self.render_frame, text="Render", command=self.controller.render)
        self.renderer_log_text = tk.scrolledtext.ScrolledText(self.render_frame, wrap=tk.WORD, height=10, width=45)
        self.renderer_log_text.configure(state='disabled')
        self.render_progressbar = ttk.Progressbar(self.render_frame, orient=tk.HORIZONTAL, length=100, mode="indeterminate")

        self.model_name_label = tk.Label(self.finetune_frame, text="Model Name: ", pady=10)
        self.model_name = tk.Entry(self.finetune_frame)
        # self.premodel_name_static = tk.Label(self.finetune_frame, text="", pady=10)
        # self.premodel_name = tk.Label(self.finetune_frame, text="", bg=controller['bg'], pady=10)
        # self.premodel_button = tk.Button(self.finetune_frame, text="...",
        #                                  command=self.controller.load_pretrained_model)
        self.output_folder_static1 = tk.Label(self.finetune_frame, text="", pady=10)
        self.output_folder1 = tk.Label(self.finetune_frame, text="", bg=controller['bg'], pady=10)
        self.output_folder_button1 = tk.Button(self.finetune_frame, text="...",
                                               command=self.controller.load_synth_output_folder)
        self.output_log_label1 = tk.Label(self.finetune_frame, text="", pady=10)
        self.output_log_button1 = tk.Button(self.finetune_frame, text="...",
                                            command=self.controller.load_finetune_log_file)
        self.output_log_name1 = tk.Label(self.finetune_frame, text="", bg=controller['bg'], pady=10)
        self.gpu_label = tk.Label(self.finetune_frame, text="", pady=10)
        self.gpu_static = tk.Label(self.finetune_frame, text="", bg=controller['bg'])
        self.finetune_kill_button = tk.Button(self.finetune_frame, text="Kill Training Thread",
                                              command=self.controller.finetune_kill_thread)
        self.finetune_default_button = tk.Button(self.finetune_frame, text="Default Settings",
                                                 command=self.controller.set_default_finetune)
        self.finetune_button = tk.Button(self.finetune_frame, text="Train", command=self.controller.finetune)
        self.finetune_log_text = tk.scrolledtext.ScrolledText(self.finetune_frame, wrap=tk.WORD, height=10, width=45)
        self.finetune_log_text.configure(state='disabled')
        self.finetune_progressbar = ttk.Progressbar(self.finetune_frame, orient=tk.HORIZONTAL, length=100,
                                                    mode="indeterminate")
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(3, weight=1)

        self.render_frame_title.grid(row=1, column=1, sticky='W')
        self.finetune_frame_title.grid(row=1, column=2, sticky='W')
        self.render_frame.grid(row=2, column=1, sticky='N')
        self.finetune_frame.grid(row=2, column=2, sticky='N')

        self.template_name_static.grid(row=0, column=0, sticky='W')
        self.template_name_button.grid(row=0, column=1)
        self.template_name.grid(row=0, column=2, sticky='W')

        self.renderer_name_static.grid(row=1, column=0, sticky='W')
        self.renderer_name_button.grid(row=1, column=1)
        self.renderer_name.grid(row=1, column=2, sticky='W')

        self.output_folder_static.grid(row=2, column=0, sticky='W')
        self.output_folder_button.grid(row=2, column=1)
        self.output_folder.grid(row=2, column=2, sticky='W')

        self.output_log_label.grid(row=3, column=0, sticky='W')
        self.output_log_button.grid(row=3, column=1)
        self.output_log_name.grid(row=3, column=2, sticky='W')

        self.iterations_number_label.grid(row=4, column=0, sticky='W')
        self.iterations_number.grid(row=4, column=2, sticky='W')

        self.render_monitor_progress_checkbox.grid(row=5, column=0, sticky='W')
        self.render_default_button.grid(row=5, column=1)
        self.render_button.grid(row=5, column=2)
        self.renderer_log_text.grid(row=6, columnspan=3)
        self.render_progressbar.grid(row=7, columnspan=3)

        self.model_name_label.grid(row=0, column=0, sticky='W')
        self.model_name.grid(row=0, column=2, sticky='W')

        # self.premodel_name_static.grid(row=2, column=0, sticky='W')
        # self.premodel_button.grid(row=2, column=1)
        # self.premodel_name.grid(row=2, column=2, sticky='W')

        self.output_folder_static1.grid(row=3, column=0, sticky='W')
        self.output_folder_button1.grid(row=3, column=1)
        self.output_folder1.grid(row=3, column=2, sticky='W')

        self.output_log_label1.grid(row=4, column=0, sticky='W')
        self.output_log_button1.grid(row=4, column=1)
        self.output_log_name1.grid(row=4, column=2, sticky='W')

        self.gpu_label.grid(row=5, column=0, sticky='W')
        self.gpu_static.grid(row=5, column=2, sticky='W')

        self.finetune_kill_button.grid(row=6, column=0, sticky='W')
        self.finetune_default_button.grid(row=6, column=1)
        self.finetune_button.grid(row=6, column=2)
        self.finetune_log_text.grid(row=7, columnspan=3)
        self.finetune_progressbar.grid(row=8, columnspan=3)
        self.update_labels()

    def render_set_defaults(self):
        self.set_entry_text(self.iterations_number, "100000")
        self.render_monitor_progress_checkbox.select()

    def finetune_set_defaults(self):
        # self.set_entry_text(self.gpu, "-1")
        self.set_entry_text(self.model_name, "my_new_model")

    def set_entry_text(self, item, text):
        item.delete(0, tk.END)
        item.insert(0, text)
        return

    def set_finetune_log_text(self, name, value, iter):
        msg = "{}, {}: {}".format(name, iter, value)
        fully_scrolled_down = self.finetune_log_text.yview()[1] == 1.0
        self.finetune_log_text.configure(state='normal')
        self.finetune_log_text.insert(tk.END, msg + "\n")
        if fully_scrolled_down:
            self.finetune_log_text.see("end")
        self.finetune_log_text.configure(state='disabled')

    def set_renderer_log_text(self, msg):
        fully_scrolled_down = self.renderer_log_text.yview()[1] == 1.0
        self.renderer_log_text.configure(state='normal')
        self.renderer_log_text.insert(tk.END, msg + "\n")
        if fully_scrolled_down:
            self.renderer_log_text.see("end")
        self.renderer_log_text.configure(state='disabled')

    def update_render_progress_bar(self, active):
        if active:
            self.render_progressbar.step(10)
        else:
            self.render_progressbar.stop()

    def update_finetune_progress_bar(self, active):
        if active:
            self.finetune_progressbar.step(10)
        else:
            self.finetune_progressbar.stop()

    def update_labels(self):
        template_file_str = self.controller.get_template_model_file_name()
        renderer_file_str = self.controller.get_renderer_file_name()
        output_folder_str = self.controller.get_synth_output_folder_name()
        render_log_file_str = self.controller.get_renderer_log_file_name()
        finetune_log_file_str = self.controller.get_finetune_log_file_name()
        # pretrained_model_str = self.controller.get_pretrained_stormnet_path()
        gpu_id_str = str(self.controller.get_gpu_id())
        self.template_name_static.config(text="Template Model File: ")
        self.template_name.config(text=template_file_str)
        self.renderer_name_static.config(text="Renderer: ")
        self.renderer_name.config(text=renderer_file_str)
        self.output_folder_static.config(text="Synthesized Data Output Folder: ")
        self.output_folder.config(text=output_folder_str)
        self.output_log_label.config(text="Renderer Log File: ")
        self.output_log_name.config(text=render_log_file_str)
        # self.premodel_name_static.config(text="Pretrained Model: ")
        # self.premodel_name.config(text=pretrained_model_str)
        self.output_folder_static1.config(text="Synthesized Data Output Folder: ")
        self.output_folder1.config(text=output_folder_str)
        self.output_log_label1.config(text="Training Log File: ")
        self.output_log_name1.config(text=finetune_log_file_str)

        self.gpu_label.config(text="GPU ID to use (-1 for CPU): ")
        self.gpu_static.config(text=gpu_id_str)

def fill_structures(orig_names, orig_data, template_file_name, mode="MNI", force_transform=False):
    # cow = ps.register_point_cloud("center_of_world", np.zeros((1, 3)), radius=0.005, point_render_mode="sphere")
    # cow.add_color_quantity("color", np.array([[0, 0, 0]]), enabled=True)
    names = orig_names
    if mode == "MNI":
        scale = 200.
        XYZ_head = MNI.load_raw_MNI_data("./resource/MNI_templates/xyzallHEM", "head", resource_folder="./resource")
        avg_head = ps.register_point_cloud("avg_head", XYZ_head, radius=0.0002, point_render_mode="quad")
        col = np.zeros_like(XYZ_head)
        col[:, 0] = 1
        avg_head.add_color_quantity("color", col, enabled=True)
        XYZ_brain = MNI.load_raw_MNI_data("./resource/MNI_templates/xyzallBEM.npy", "brain", resource_folder="./resource")
        avg_brain = ps.register_point_cloud("avg_brain", XYZ_brain, radius=0.0002, point_render_mode="quad")
        col = np.zeros_like(XYZ_brain)
        col[:, 1] = 1
        avg_brain.add_color_quantity("color", col, enabled=True)
        pc_name = str(template_file_name) + "(MNI)"
        data = orig_data
        radius=0.005
        if force_transform:
            names_copy = orig_names.copy()
            data_copy = orig_data.copy()
            sensors = geometry.project_sensors_to_MNI([[names_copy, data_copy]], transform_anchors=True)
            names = sensors[0][0]
            data = sensors[0][1]
            radius=0.005
    elif mode == "STORM-Net":
        scale = np.abs(orig_data.max() - orig_data.min()) * 2
        data = geometry.to_standard_coordinate_system(orig_names, orig_data)
        pc_name = str(template_file_name)+ "(normalized)"
        radius=0.005
    elif mode == "Original":
        scale = np.abs(orig_data.max() - orig_data.min()) * 2
        data = orig_data
        pc_name = str(template_file_name)
        radius=0.005
    else:
        raise NotImplementedError
    v_aabb, e_aabb, c_aabb = geometry.get_curve_network_aabb(scale=scale)
    aabb = ps.register_curve_network("aabb", v_aabb, e_aabb, radius=0.005)
    aabb.add_color_quantity("color", c_aabb, defined_on='edges', enabled=True)
    pc = ps.register_point_cloud(pc_name,
                                 data,
                                 radius=radius)
    anchor_mask = np.isin(np.array(names), np.array(config.all_possible_anchor_names)).squeeze()
    sticker_mask = np.isin(np.array(names), np.array(config.physical_sticker_names)).squeeze()
    colors = np.ones_like(data)
    colors[anchor_mask] = 0
    colors[sticker_mask] = np.array([0, 1, 0])
    pc.add_color_quantity("colors",
                            colors,
                            enabled=True)