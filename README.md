# STORM-Net: Simple and Timely Optode Registration Method for Functional Near-Infrared Spectroscopy (FNIRS)
All files in the repository are an implementation of the original manuscript and research ([preprint](https://doi.org/10.1101/2020.12.29.424683)).
## Introduction
This application is designed to provide an accurate estimation of the position of an fNIRS probing cap on a participant’s head, based on a short video. In other words, given a short video of the participant wearing the fNIRS cap, the application outputs the coordinates of every point of interest on the cap in (a statistical) MNI coordinate system (or, if required, in the original coordinates system that can be later transformed to MNI using external tools such as [SPM fNIRS](https://www.nitrc.org/projects/spm_fnirs/)).

There are 3 modes of operation:
1. A GUI that allows manual annotation of the data / supervising the automatic method - this is recommended for first time users.
2. A command-line interface which is a ready-to-use end-to-end script that performs calibration given a video file - use this when you are comfortable with the GUI and its results.
3. An experimental mode that allows reproducing all results from the original manuscript.

The repository also contains:
- Python scripts for training the neural networks discussed in the paper. Description of how to do this is down below. 
- A synthetic data generator implemented in [Unity](https://unity.com/).

### Application dependencies:
- Python version >= 3.6, but <= 3.8 (x64)
- [environment.yml](environment.yml) contains all python library dependencies (we suggest using a seperate environment such as conda or virtualenv):\
      `conda install -f environment.yml`
      
- Neural-network model files which can be downloaded from [here](https://www.cs.tau.ac.il/~yotamerel/models/storm_models.zip). \
      Place the files under the [models](CapCalibrator/models) folder (after extracting).
- Precompiled binaries for the renderer which can be downloaded from here: [windows](https://www.cs.tau.ac.il/~yotamerel/precompiled_binaries/DataSynth/windows_build.zip), linux and mac users must compile the renderer from source using [Unity](https://unity.com/) 2019.3 or higher.
- Hardware: Although not necessary, the automatic anotation performes significantly faster when a compliant GPU is available (by 2 orders of magnitude). We recommend using a GPU if fast calibration times are needed.

Note: tested on Windows 10 and Ubuntu 18.04. Should work for Mac as well.

## Quick installation guide
### Step 1: Clone this repository to get a copy of the code to run locally.

Clone the repository by downloading it [directly](https://github.com/yoterel/STORM-Net/archive/master.zip) or by using the git command line tool:\
`git clone https://github.com/yoterel/STORM-Net.git`

### Step 2: Navigate to the STORM-Net directory, then create a virtual environment using conda.

We recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for this, but you can also [Install Anaconda](https://www.anaconda.com/products/individual/get-started) if needed, then create an environment using the environment.yml file in this repository:

`conda env create -n env -f environment.yml`

Note: For dlib (part of the requirements in the environment file), you must have some modern C++ able compiler like [visual studio](https://visualstudio.microsoft.com/) or gcc installed on your computer.

### Step 3: Download all pre-trained neural network models.

Download from [here](https://www.cs.tau.ac.il/~yotamerel/models/storm_models.zip), and place them under the [models](CapCalibrator/models) folder (after extracting).

### Step 4: Download all precompiled binaries for the renderer.
Download from here: [windows](https://www.cs.tau.ac.il/~yotamerel/precompiled_binaries/DataSynth/windows_build.zip), linux and mac users must compile the renderer from source using [Unity](https://unity.com/) 2019.3 or higher.

### Step 5: Run STORM-Net in gui mode.
First remember to activate the environment you created
`conda activate env`
Then navigate to [main.py](CapCalibrator/main.py), and run:\
`python main.py --mode gui`

Note: in Linux you might need to unset pythonpath (after step 7) before the application can be run successfully (step 8):\
`unset PYTHONPATH`

## How to use the different modes

The file [main.py](CapCalibrator/main.py) is the entry point of the application. Some common use cases are shown below:

`python main.py --mode gui`

`python main.py --mode auto --video path_to_video_file --template path_to_template_file`

`python main.py --mode experimental --video path_to_video_folder --template path_to_template_file --gt path_to_ground_truth_measurements_folder`

The mode "gui" indicates to the application that the user wants to use the GUI and supervise the process of annotation and calibration and to correct it if needed. This is recommended when possible. Note the GUI contains other useful functions such as viewing a template model and finetunning the neural networks.

The mode "auto" indicates to the application that the user wants it to automatically annotate the video without any supervision. This is recommended for live sessions and when the system was oberved to perform well with a certain template model. Note that using this mode the application requires two additional paramters which are the path to the raw video file and to a template model file.

The mode "experimental" indicates to the application that the user wants to reproduce all results in the original paper. Note this requires the original dataset used and is available upon request from the corrosponding author.

For all command line options, see:

`python main.py --help`

## The template model file

The application uses a pre-measured template model file as an input. An example of such file is located under the [example_models](example_models) directory.
This template model was obtained using a 3D digitizer, but can similarly be obtained in any way as long as the measurements are accurate to a satisfactory degree.
It is strongly recommended to create your own template model file **per physical cap model you are using in your lab**, this will lead to best accuracies.
The exact format of this file is now specified.
The file is a csv space delimited file, where each row contains 4 values:
1. An index of the optode (an integer numerical value representing the index of this optode. Starts from 0. any other value (strings, etc) will not be calibrated but still appear in the output file in the same standard coordiante system as the rest of the optodes).
2. 3 numerical values: X, Y, Z representing the location of this optode (note: values must be supplied in cm or inch).
The coordinate system these values are supplied in are not improtant, as they are transformed internally to a standard right-handed system, and output coordinates are in this new cooridnate system (described below).

## Using a new cap

After creating a template model for a new cap, the offline step is required to be performed.
To do this, follow the steps below:

1. Create synthetic data using the GUI or the [render script](DataSynth/render.py) (notice this script requires the template model file path and the renderer executable path as input). We recommend usings a minimum of 100000 iterations:\
   `python render.py path_to_template_model_file path_to_output_folder --exe path_to_renderer_executable --log path_to_output_log --iterations 100000`
   
   For all command line options when rendering see:
   
   `python render.py --help`
   
2. Train the network on the images using the GUI or the [train script](CapCalibrator/train.py). We recommend using a gpu to speed up the training process:\
   `python train.py my_new_model_name path_to_synthetic_data_folder --gpu_id 2`
   
   For all command line options see:
   
   `python train.py --help`
   
   When training is done, a model file will be availble in the [models](CapCalibrator/models) directory.
   
   Note: we strongly suggest to train until validation loss reaches atleast 0.1 - do not stop before this.
   
3. Use this model in the arguments supplied to the GUI / automatic calibration tools.

## List of mandatory optodes in template file

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
["cz", "nosebridge", "inion", "rightear", "leftear", "fp1", "fp2", "fpz", "f7", "f8", "f3", "f4", "c3", "c4", "t3", "t4", "pz", "p3", "p4", "t5", "t6", "o1", "o2"]

Notes:
- Use these names (exactly, without double quoutes) as the first field for them to be parsed correctly.
- The stickers can be placed anywhere **on the cap** as long as the three frontal ones are not colinear, but we recommend using fp1, fpz + 1cm, fp2, cz as their location.
- middle_triangle: we actually used a location 1 cm above fpz in our experiments (can be seen marked by a green sticker in the example video) - this eliminated the risk of coliniearity (but it is not mandatory).


## The standard coordiante system

This application outputs data in the following coordiante system (notice it is right-handed):
- X axis is from left to right ear
- Y axis is from back to front of head
- Z axis is from bottom to top of head
- The origin is defined by (x,y,z) = ((lefteye.x+righteye.x) / 2, cz.y, (lefteye.z+righteye.z) / 2)
- Scale is cm. If "CZ" is too close to origin in terms of cm, the code scales it to cm (by assuming it is measured in inches).
