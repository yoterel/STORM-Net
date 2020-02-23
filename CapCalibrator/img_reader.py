import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import scipy.io as sio
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text.stem) ]

frame_dir = Path("E:/globus_data/Frames")
label_dir = Path("E:/globus_data/New Labels/PixelLabelData_1")
db_dir = Path("E:/globus_data/db_segmentation/Labels")
for file in db_dir.glob("*"):
    file_name = file.stem
    new_file_name = file.stem.replace("_label","")
    new_path = Path.joinpath(file.parent, new_file_name + ".png")
    file.rename(new_path)
# count = 0
# frames = sorted(frame_dir.glob('*'))
# labels = sorted(label_dir.glob('*'))
# labels.sort(key=natural_keys)
# for i, file in enumerate(frames):
#         new_path = Path.joinpath(labels[i].parent, file.stem+"_label.png")
#         labels[i].rename(new_path)


