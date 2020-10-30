from pathlib import Path
import video
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import predict
from PIL import Image, ImageTk
import numpy as np
import file_io
import queue
import threading


class AnnotationPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        h, w = self.controller.get_frame_size()
        self.canvas = tk.Canvas(self, height=h, width=w, bg="#263D42")
        img = ImageTk.PhotoImage(master=self, image=self.controller.get_cur_frame())
        self.canvas.img = img  # or else image gets garbage collected
        self.canvas.create_image(0, 0, anchor="nw", image=img, tag="image")
        self.canvas.bind("<ButtonPress-1>", self.controller.save_coords)
        self.canvas.bind("<ButtonPress-2>", self.controller.zero_coords)
        self.canvas.bind("<ButtonPress-3>", self.controller.next_coords)
        pad_x = 0
        pad_y = 5
        button_width = 5
        nextFrame = tk.Button(self, text="Next Frame", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                              bg="#263D42", command=self.controller.next_frame)
        prevFrame = tk.Button(self, text="Previous Frame", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                              bg="#263D42", command=self.controller.prev_frame)
        nextVideo = tk.Button(self, text="Next Video", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                              bg="#263D42", command=self.controller.next_video)
        prevVideo = tk.Button(self, text="Previous Video", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                              bg="#263D42", command=self.controller.prev_video)
        shiftVideoF = tk.Button(self, text="Shift Forward", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                                bg="#263D42", command=self.controller.shift_video_f)
        shiftVideoB = tk.Button(self, text="Shift Backward", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                                bg="#263D42", command=self.controller.shift_video_b)
        loadSession = tk.Button(self, text="Load Session", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                                bg="#263D42", command=self.controller.load_session)
        saveSession = tk.Button(self, text="Save Session", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                                bg="#263D42", command=self.controller.save_session)
        autoAnnotate = tk.Button(self, text="Auto Annotate", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                                bg="#263D42", command=self.controller.auto_annotate)
        doneButton = tk.Button(self, text="Done", width=button_width, padx=pad_x, pady=pad_y, fg="white",
                               bg="#263D42", command=self.controller.destroy)
        self.data_panel = tk.Frame(self, bg="white")
        # self.update_labels()
        loadSession.grid(row=0, column=0, sticky="w"+"e")
        saveSession.grid(row=0, column=1, sticky="w"+"e")
        prevVideo.grid(row=0, column=2, sticky="w"+"e")
        nextVideo.grid(row=0, column=3, sticky="w"+"e")
        prevFrame.grid(row=0, column=4, sticky="w"+"e")
        nextFrame.grid(row=0, column=5, sticky="w"+"e")
        shiftVideoB.grid(row=0, column=6, sticky="w"+"e")
        shiftVideoF.grid(row=0, column=7, sticky="w"+"e")
        autoAnnotate.grid(row=0, column=8, sticky="w" + "e")
        doneButton.grid(row=0, column=9, sticky="w"+"e")
        self.data_panel.grid(row=0, column=10, rowspan=10)
        self.canvas.grid(row=1, columnspan=10, rowspan=9)

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
        self.model = None
        self.graph = None
        if self.args.mode == "semi-automatic" or self.args.mode == "experimental":
            model_name = args.u_net
            model_dir = Path("models")
            model_full_name = Path.joinpath(model_dir, model_name)
            self.model, self.graph = file_io.load_semantic_seg_model(str(model_full_name))
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
        self.take_async_action(self.prep_vid_to_frames_packet())

    def process_queue(self):
        try:
            msg = self.queue.get(False)
            if msg[0] == "video_to_frames":
                self.frames, self.indices, my_dict = msg[1:]
                if my_dict:
                    self.db[self.get_cur_video_name()] = [my_dict]
                else:
                    if self.get_cur_video_name() not in self.db.keys():
                        data = np.zeros((1, 10, 14))
                        my_dict = {"data": data,
                                   "label": np.array([0, 0, 0]),
                                   "frame_indices": self.indices}
                        self.db.setdefault(self.get_cur_video_name(), []).append(my_dict)
                self.panels[AnnotationPage].update_canvas()
                self.panels[AnnotationPage].update_labels()
            elif msg[0] == "shift_video":
                self.frames, self.indices = msg[1:]
                self.db[self.get_cur_video_name()][self.shift]["frame_indices"] = self.indices
                print("new indices:", self.indices)
                self.panels[AnnotationPage].update_canvas()
                self.panels[AnnotationPage].update_labels()
            # Show result of the task if needed
            self.panels[ProgressBarPage].show_progress(False)
            self.show_panel(AnnotationPage)
            self.enable_menu()
        except queue.Empty:
            self.after(100, self.process_queue)

    def take_async_action(self, msg):
        msg_dict = {"video_to_frames": "Selecting frames & predicting landmarks. This might take some time if GPU is not being used.\nUsing GPU: {}".format(predict.is_using_gpu()),
                    "shift_video": "Selecting different frame..."}
        self.disbale_menu()
        self.show_panel(ProgressBarPage)
        self.panels[ProgressBarPage].show_progress(True)
        self.panels[ProgressBarPage].update_status_label(msg_dict[msg[0]])
        ThreadedTask(self.queue, msg).start()
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

    def prep_vid_to_frames_packet(self, perform_pred=None):
        path = self.paths[self.cur_video_index]
        if not perform_pred:
            if self.args.mode == "semi-auto":
                perform_pred = True
                name = self.get_cur_video_name()
                if name in self.db.keys():
                    if self.db[self.get_cur_video_name()][self.shift]["data"] != np.zeros((1, 10, 14)):
                        perform_pred = False
            else:
                perform_pred = False
        return ["video_to_frames", path, perform_pred, self.model, self.graph, self.args]

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

    def save_coords(self, event):
        current_video = self.get_cur_video_name()
        self.db[current_video][self.shift]["data"][0, self.cur_frame_index, self.cur_sticker_index:self.cur_sticker_index+2] = event.x, 540-event.y
        self.go_to_next_coord()

    def zero_coords(self, event):
        current_video = self.get_cur_video_name()
        self.db[current_video][self.shift]["data"][0, self.cur_frame_index, self.cur_sticker_index:self.cur_sticker_index+2] = 0, 0
        self.go_to_next_coord()

    def next_coords(self, event):
        self.go_to_next_coord()

    def auto_annotate(self):
        self.cur_frame_index = 0
        self.cur_sticker_index = 0
        self.take_async_action(self.prep_vid_to_frames_packet(True))

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
        current_video = self.get_cur_video_name()
        current_indices = self.db[current_video][self.shift]["frame_indices"]
        print("current indices:", current_indices)
        new_indices = current_indices.copy()
        new_indices[self.get_cur_frame_index()] += 1
        self.take_async_action(["shift_video", self.paths, self.cur_video_index, new_indices])

    def shift_video_b(self):
        current_video = self.get_cur_video_name()
        current_indices = self.db[current_video][self.shift]["frame_indices"]
        print("current indices:", current_indices)
        new_indices = current_indices.copy()
        new_indices[self.get_cur_frame_index()] -= 1
        self.take_async_action(["shift_video", self.paths, self.cur_video_index, new_indices])

    def next_frame(self):
        if self.cur_frame_index < 9:
            self.cur_frame_index += 1
            self.cur_sticker_index = 0
            self.panels[AnnotationPage].update_canvas()
            self.panels[AnnotationPage].update_labels()

    def prev_frame(self):
        if self.cur_frame_index > 0:
            self.cur_frame_index -= 1
            self.cur_sticker_index = 0
            self.panels[AnnotationPage].update_canvas()
            self.panels[AnnotationPage].update_labels()

    def load_session(self):
        filename = filedialog.askopenfilename(initialdir="./data", title="Select Session File")
        if not filename:
            return
        self.db = file_io.load_from_pickle(filename)
        self.take_async_action(self.prep_vid_to_frames_packet())

    def save_session(self):
        f = filedialog.asksaveasfile(initialdir="./data", title="Select Session File", mode='wb')
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        file_io.dump_to_pickle(Path(f.name), self.db)


class ThreadedTask(threading.Thread):
    def __init__(self, queue, msg):
        threading.Thread.__init__(self)
        self.queue = queue
        self.msg = msg

    def run(self):
        if self.msg[0] == "video_to_frames":
            self.handle_video_to_frames()
        elif self.msg[0] == "shift_video":
            self.handle_shift_video()

    def handle_video_to_frames(self):
        path, perform_pred, preloaded_model, graph, args = self.msg[1:]
        frames, indices = video.video_to_frames(path, dump_frames=True)
        name = path.parent.name + "_" + path.name
        my_dict = {}
        if perform_pred:
            data = predict.predict_keypoints_locations(frames,
                                                       args,
                                                       name,
                                                       is_puppet=False,
                                                       save_intermed=False,
                                                       preloaded_model=preloaded_model,
                                                       graph=graph)
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


def annotate_videos(video_path, args):  # contains GUI mainloop
    if args.mode == "experimental":
        special_db = Path.joinpath(Path("data"), "telaviv_db.pickle")
        new_db = file_io.load_full_db(special_db)
    else:
        new_db = file_io.load_full_db()
    paths = []
    if Path.is_file(video_path):
        paths.append(video_path)
    else:
        for file in video_path.glob("**/*.MP4"):
            paths.append(file)

    app = GUI(new_db, paths, args)
    app.mainloop()
    return app.get_db()


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
