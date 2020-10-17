import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
import argparse
from pathlib import Path
import numpy as np
from draw import Arrow3D, plot_3d_pc
from geometry import to_standard_coordinate_system
from file_io import read_template_file



selected = 0
LARGE_FONT = ("Verdana", 12)


class GUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Experiment Viewer")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (MainMenu, PageOne, ExperimentViewer):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(ExperimentViewer)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class MainMenu(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="fNIRS Calibration", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button = ttk.Button(self, text="Dummy Page 1",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

        button3 = ttk.Button(self, text="Experiment Viewer",
                             command=lambda: controller.show_frame(ExperimentViewer))
        button3.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Dummy Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Main Menu",
                             command=lambda: controller.show_frame(MainMenu))
        button1.pack()


class ExperimentViewer(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # label = tk.Label(self, text="Graph Page!", font=LARGE_FONT)
        # label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(MainMenu))
        button1.pack()
        f = plt.Figure(figsize=(7, 7), dpi=100)
        a = f.add_subplot(111, projection='3d')
        a.view_init(60, -470)
        # b = quiver(X, Y, Z, U, V, W, **kwargs)
        args = parse_arguments()
        names, data, format = read_template_file(args.template)
        data = data[0]  # read only first session
        names = names[0]  # read only first session
        if format == "telaviv":
            if args.use_sensor2:
                data = data[:, 0, :] - data[:, 1, :]
            else:
                data = data[:, 0, :]
        data = to_standard_coordinate_system(names, data)
        if not args.only_fiducials:
            spiral_index = names.index(0)
            data = data[spiral_index:, :]
            names = names[spiral_index:]
        plot_3d_pc(a, data, selected, names)
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.mpl_connect("key_press_event", lambda event: key_press_callback(event, data, names))
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


def key_press_callback(event, data, names):
    global selected
    total_data_points = len(data)
    ax = event.inaxes
    # ax.autoscale(enable=False, axis='both')
    # Rotational movement
    elev = ax.elev
    azim = ax.azim
    if event.key == "w":
        elev += 10
    elif event.key == "s":
        elev -= 10
    elif event.key == "d":
        azim += 10
    elif event.key == "a":
        azim -= 10
    elif event.key == "right":
        if selected < total_data_points-1:
            selected += 1
        else:
            return
    elif event.key == "left":
        if selected > 0:
            selected -= 1
        else:
            return
    elif event.key == "backspce":
        selected = 0
    else:
        return
    ax.cla()
    ax.view_init(elev=elev, azim=azim)
    plot_3d_pc(ax, data, selected, names)
    ax.figure.canvas.draw()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Renders training images for fNIRS alighment')
    parser.add_argument("template", help="Path to raw template file. For exact format see documentation.")
    parser.add_argument("--use_sensor2", action='store_true', help="sensor2 results will be subtracted from sensor1 for more accuracy.")
    parser.add_argument("--only_fiducials", action="store_true", help="viewer just shows specific fiducials.")
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    cmd_line = "./../example_models/telaviv_experiment/maya.txt --use_sensor2 --only_fiducials".split()
    args = parser.parse_args(cmd_line)
    args.template = Path(args.template)
    print(args.template)
    return args


if __name__ == "__main__":
    app = GUI()
    app.mainloop()
