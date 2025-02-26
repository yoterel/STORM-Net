[![release](https://github.com/yoterel/STORM-Net/actions/workflows/main.yml/badge.svg)](https://github.com/yoterel/STORM-Net/actions/workflows/main.yml)

# STORM-Net
Simple and Timely Optode Registration Method for Functional Near-Infrared Spectroscopy (FNIRS)


<table>
  <tr>
    <td> <img src="https://github.com/yoterel/STORM-Net/blob/master/resource/img1.png"  alt="1" width = 400px height = 400px ></td>
    <td><img src="https://github.com/yoterel/STORM-Net/blob/master/resource/img2.png" alt="2" width = 400px height = 400px></td>
   </tr>
</table>

All files in the repository are an implementation of the original manuscript and research ([preprint](https://doi.org/10.1101/2020.12.29.424683)).

# Introduction
This application is designed to provide an accurate estimation of the position of points of interest (usually, sensors) on a fNIRS probing cap mounted on a subject’s head, based on a short video. This is needed because the cap is almost never placed in perfect alignment with the actual position it was intended for (mostly due to human error, but also because of differences in skull structures and cap deformations).

In other words, given a short video of the subject wearing the fNIRS cap, the application outputs the coordinates of every point of interest on the cap in (a statistical) MNI coordinate system (or, if required, in the original coordinates system that can be later transformed to MNI using external tools such as [SPM fNIRS](https://www.nitrc.org/projects/spm_fnirs/)).

There are 3 modes of operation:
1. A GUI that allows manual annotation of the data / supervising the automatic method - this is recommended for first time users.
2. A command-line interface which is a ready-to-use end-to-end script that performs calibration given a video file - use this when you are comfortable with the GUI and its results.
3. An experimental mode that allows reproducing all results from the original manuscript.

The repository also contains:
- Python scripts for training the neural networks discussed in the paper. Description of how to do this is down below. 
- A synthetic data generator implemented in [Unity](https://unity.com/).

# Quick installation guide
Note: Ubuntu 18.04, and WSL (Windows Subsystem for Linux). Windows and Mac aren't currently supported. The following assumes "git" and "cmake" are installed and accessible e.g.:

`sudo apt-get install git` \
`sudo apt-get install cmake`

### Step 1: Clone this repository to get a copy of the code to run locally.

Clone the repository by downloading it [directly](https://github.com/yoterel/STORM-Net/archive/master.zip) or by using the git command line tool:\
`git clone https://github.com/yoterel/STORM-Net.git`

Navigate into it:\
`cd STORM-Net`

### Step 2: Navigate to the STORM-Net directory, then create a virtual environment using micromamba.

We recommend installing [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for this. After installing it, use:

`micromamba create -n storm python=3.10 -c conda-forge`

Then activate the environment and install the requirements:
`micromamba activate storm` \
`pip install -r requirements.txt`

Note: if dlib failed to build, it is likely due to cmake not being accessible. 

### Step 3: Download all pre-trained neural network models.

Download from [here](https://osf.io/3j6u2/download), and place them under the [models](CapCalibrator/models) folder (after extracting, e.g. using unzip).

### Step 4: Download all precompiled binaries for the renderer.

Download, extract and change permission if needed:
`cd DataSynth`\
`wget -O linux_build.zip https://osf.io/56a28/download`\
`unzip linux_build.zip`\
`chmod -R 777 linux_build`\
`cd ..`

### Step 5: Run STORM-Net in gui mode.
First remember to activate the environment you created
`micromamba activate storm`
Then navigate to [main.py](CapCalibrator/main.py):\
`cd CapCalibrator`\

And run:\
`python main.py --mode gui`\

Or if you have a supported GPU:\
`python main.py --mode gui --gpu_id 0`

## Modes of operation

The file [main.py](CapCalibrator/main.py) is the entry point of the application. Some common use cases are shown below:

`python main.py --mode gui`

`python main.py --mode auto --video path_to_video_file --template path_to_template_file`

`python main.py --mode experimental --video path_to_video_folder --template path_to_template_file --gt path_to_ground_truth_measurements_folder`

The mode "gui" indicates to the application that the user wants to use the GUI and supervise the process of annotation and registration and to correct it if needed. This is recommended when possible. Note the GUI contains other useful functions such as viewing a template model and finetunning the neural networks.

The mode "auto" indicates to the application that the user wants it to automatically annotate the video without any supervision. This is recommended for live sessions and when the system was oberved to perform well with a certain template model. Note that using this mode the application requires two additional paramters which are the path to the raw video file and to a template model file.

The mode "experimental" indicates to the application that the user wants to reproduce all results in the original paper. Note this requires the original dataset used and is available upon request from the corrosponding author.

For all command line options, see:

`python main.py --help`

# Usage guide

In a nutshell, you will need to perform the "offline step" (slow) prior to performing registration (once per phsyical cap), and an "online step" (fast) every time you want to coregister, e.g. a subject was mounted with the cap and you want to coregister.

## Offline step

The offline step (and infact, the online step too) rely on a pre-measured template model file as an input. An example of such file is located under the [example_models](example_models) directory. This template model was obtained using a 3D digitizer, but can similarly be obtained in any way as long as the measurements are accurate to a satisfactory degree (e.g. by 3D scanning, or taking a video and running [COLMAP](https://colmap.github.io/)).
It is strongly recommended to create your own template model file **per physical cap model you are using in your lab**, this will lead to best accuracies.

for the exact format, and list of the minimum neccessary points that are required in this file, see below in the section "Template Model File".

### Rendering

After creating a template model for a new cap, the offline step first stage can be performed (rendering synthetic data).
To create synthetic data, you can either use the GUI -> "Offline Step" -> "Render" (set the appropriate fields, we recommend usings a minimum of 100000 iterations) or use the [render script](DataSynth/render.py) directly. Notice this script requires the template model file path and the renderer executable path as input:\
   `python render.py path_to_template_model_file path_to_output_folder --exe path_to_renderer_executable --log path_to_output_log --iterations 100000`
   
   For all command line options when rendering see:
   
   `python render.py --help`

### Training

Following rendering, the second stage of the offline step can be performed (training STORM-Net).
To train the network on the renderings you produced from the previous step, use the GUI -> "Offline Step" -> "Train" (set the appropriate fields) or use the [train script](CapCalibrator/torch_train.py) directly. We recommend using a gpu to speed up the training process:\
   `python torch_train.py my_new_model_name path_to_synthetic_data_folder --gpu_id 2`
   
   For all command line options see:
   
   `python torch_train.py --help`
   
 When training is done, a model file will be availble in the [models](CapCalibrator/models) directory.
   
 Note: we strongly suggest to train until validation loss reaches atleast 0.2 - do not stop before this.
   
## Online step

Now that we have a trained model, we can take a video of the subject wearing the cap.
The video must be taken using the camera path described in the paper, see [example video](https://github.com/yoterel/STORM-Net/blob/master/example_videos/example_video.mp4) for how a proper video should be taken). Notice the video must have a 16∶9 aspect ratio (for example, 1920x1080 resolution) and must be taken horizontally.

in the GUI -> "Online Step", select File from the menu, and load the required inputs one by one (Storm-Net model you trained, the video itself, and the template model recorded for the offline step). STORM-Net will automatically select 10 frames for you, and you can either automatically annotate the facial landmarks and stickers, or manually do so, for each frame (use the arrow keys to scroll between the frames, and mouse to annotate). Note that marking the stickers does not need to have any order, i.e. CAP1, CAP2, CAP3, CAP4 are some sticker locations that can be seen in a frame, and their order doesn't matter (in the frame itself, or between frames).

After annotation is done, click Video in the top menu, and then "co-register". You can save the result to any file.

# Template Model File

The exact format of this file is now specified.
The file is a space delimited human readable file, where each row contains 4 values:
1. A unique name of the optode (some names are reserved, see below).
2. Three numerical values: X, Y, Z representing the location of this optode (note: values must be supplied in cm or inch).
The coordinate system these values are supplied in are not important, as they are transformed to STORM-Nets internal right-handed system (described below). However, the transformation to this new coordinate system relies on your measurements making physical sense, e.g. the nasion is located between the eyes, rpa is on the same side of the brain as the right eye, etc.

## List of mandatory locations in template file

The following optodes must exist in the template model file (and their locations should corrospond with real 10-20 locations as much as possible):
- "left_triangle" : the left most sticker location, use fp1 if possible. Used by the renderer.
- "middle_triangle" : the middle sticker location, use fpz location + 1cm upwards if possible. Used by the renderer.
- "right_triangle" : the right most sticker location, use fp2 if possible. Used by the renderer.
- "top" : the sticker on the top of the head, use cz if possible.
- "cz" : used to find standard coordinate system, and projecting to MNI after calibration is done (if user requires).
- "righteye" : used to find standard coordinate system and by renderer.
- "lefteye" : used to find standard coordinate system and by renderer.
- "nosetip" : used to find standard coordinate system and by renderer.
- "fp1" : used to find standard coordinate system.
- "fpz" : used to find standard coordinate system.
- "fp2" : used to find standard coordinate system.

At least **Four** (the more the better) of the following optodes must exist in the template model file if MNI projection is required (whether using our implementation of the MNI projection, **or not**):
["cz", "nz", "iz", "rpa", "lpa", "fp1", "fp2", "fpz", "f7", "f8", "f3", "f4", "c3", "c4", "t3", "t4", "pz", "p3", "p4", "t5", "t6", "o1", "o2"]

These are standard 10-20 locations. note (rpa, lpa) stands for (right, left) preauricular points.

Notes:
- Use these names (exactly, without double quoutes) as the first field for the file to be parsed correctly.
- The stickers can be placed anywhere **on the cap** as long as the three frontal ones are not colinear, but we recommend using fp1, fpz + 1cm, fp2, cz as their location.
- middle_triangle: we actually used a location 1 cm above fpz in our experiments (can be seen marked by a green sticker in the example video) - this eliminated the risk of coliniearity (but this specific location is not mandatory).
- Do not use any of these names for your sensors. They are reserved.


## STORM-Net coordinate system

This application uses the following coordiante system internally (notice it is right-handed):
- X axis is from left to right ear
- Y axis is from back to front of head
- Z axis is from bottom to top of head
- The origin is defined by (x,y,z) = ((lefteye.x+righteye.x) / 2, cz.y, (lefteye.z+righteye.z) / 2)
- Scale is cm. If "CZ" is too close to origin in terms of cm, the code scales it to cm (by assuming it is measured in inches).

## Troubleshooting issues

1. Please skim through the closed github issues to see if someone already posted a similar question.
2. For any new issue, please first open a github issue and explain your problem (we kindly request not to email any of the authors at this stage, github issues will be answered in a timely fashion).
3. If this doesn't help, the corrosponding author is available for online meetings to resolve any edge case.
