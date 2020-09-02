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
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

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
        names, data = read_template_file(args.template)
        plot_experiment(a, data)
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.mpl_connect("key_press_event", lambda event: key_press_callback(event, data))
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Renders training images for fNIRS alighment')
    parser.add_argument("template", help="Path to raw template file. For exact format see documentation.")
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    # cmd_line = "./../example_models/experiment_model.txt".split()
    args = parser.parse_args()
    args.template = Path(args.template)
    return args


def read_template_file(template_path):
    file_handle = open(str(template_path))
    file_contents = file_handle.read()
    contents_split = file_contents.splitlines()
    data = []
    names = []
    for line in contents_split:
        name, x, y, z = line.split()
        x = float(x)
        y = float(y)
        z = float(z)
        data.append(np.array([x, y, z]))
        names.append(name.lower())
    data = np.array(data)
    return names, data


def key_press_callback(event, data):
    global selected
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
        selected += 1
    elif event.key == "left":
        selected -= 1
    else:
        return
    ax.cla()
    ax.view_init(elev=elev, azim=azim)
    plot_experiment(ax, data)
    ax.figure.canvas.draw()


def plot_experiment(ax, data):
    colors = ['b'] * len(data)
    colors[selected] = 'r'
    data_min = np.min(data, axis=0)
    a = Arrow3D([data_min[0], data_min[0]+3], [data_min[1], data_min[1]],
                [data_min[2], data_min[2]], mutation_scale=10,
                lw=1, arrowstyle="-|>", color="r")
    b = Arrow3D([data_min[0], data_min[0]], [data_min[1], data_min[1]+3],
                [data_min[2], data_min[2]], mutation_scale=10,
                lw=1, arrowstyle="-|>", color="r")
    c = Arrow3D([data_min[0], data_min[0]], [data_min[1], data_min[1]],
                [data_min[2], data_min[2]+3], mutation_scale=10,
                lw=1, arrowstyle="-|>", color="r")
    d = Arrow3D([data[selected, 0], data[selected+1, 0]], [data[selected, 1], data[selected+1, 1]],
                [data[selected, 2], data[selected+1, 2]], mutation_scale=10,
                lw=1, arrowstyle="-|>", color="r")
    ax.add_artist(a)
    ax.add_artist(b)
    ax.add_artist(c)
    ax.add_artist(d)
    for i, (c, x, y, z) in enumerate(zip(colors, data[:, 0], data[:, 1], data[:, 2])):
        ax.scatter(x, y, z, marker='o', c=c)
        ax.text(x + 0.2, y + 0.2, z + 0.2, '%s' % (str(i+1)), size=6, zorder=1, color='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title('Point {} (WASD: change view, Arrows: next/previous point)'.format(selected))


if __name__ == "__main__":
    app = GUI()
    app.mainloop()
