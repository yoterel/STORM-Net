# STORM: Simple and Timely Optode Registration Method for Functional Near-Infrared Spectroscopy (FNIRS)
All files in the repository are an implementation of the original manuscript and research which is not yet published.
## Introduction
This application is designed to provide an accurate estimation of the position of an fNIRS probing cap on a participant’s head, based on a short video. In other words, given a short video of the participant wearing the fNIRS cap, the application outputs the coordiantes of every optode position in MNI coordinates (or a standard cooridnate system that can be easly transformed to MNI using external tools such as [SPM fNIRS](https://www.nitrc.org/projects/spm_fnirs/)).

It contains three tools:
1. A GUI that allows manual annotation of the data or supervising the automatic method ("semi-supervised") - this is recommended for first time users.
2. A ready-to-use end-to-end script that performs calibration given a video file - use this when you are comfortable with the GUI and its results.
3. A bundle of python scripts for finetunning (or training from scratch) the neural networks we provide - this is suggested for every new phsyical cap your lab uses. Description of how to do this is down below. Note this bundle also includes a synthetic data generator implemented in [Unity](https://unity.com/).

### Application dependencies:
- Python 3.6 or higher (required by (1), (2), and (3))
- Use the [requirments.txt](requirements.txt) file to obtain all python library dependencies (we suggest using a seperate environment such as conda or virtualenv):\
      `pip install -r requirements.txt`
      
- Neural-network model files which can be downloaded from [here](https://www.cs.tau.ac.il/~yotamerel/models/storm_models.zip). \
      Place the files under the [models](CapCalibrator/models) folder (after extracting).
- Precompiled binaries for the renderer which can be downloaded from here: [windows](https://www.cs.tau.ac.il/~yotamerel/precompiled_binaries/DataSynth/windows_build.zip), [linux](https://www.cs.tau.ac.il/~yotamerel/precompiled_binaries/DataSynth/linux_build.zip), [mac](https://www.cs.tau.ac.il/~yotamerel/precompiled_binaries/DataSynth/mac_build.zip). Note these can also be compiled from source using [Unity](https://unity.com/) 2019.3 or higher.
- Hardware: Although not necessary, the automatic anotation performes significantly faster when a compliant GPU is available (by 2 orders of magnitude). We recommend using a GPU if fast calibration times are needed.

## The template model file

The application uses a pre-measured template model file as an input. An example of such file is located under the [example_models](example_models) directory.
This template model was obtained using a 3D digitizer, but can similarly be obtained in any way as long as the measurements are accurate to a satisfactory degree.
It is strongly recommended to create your own template model file **per physical cap model you are using in your lab**, this will lead to best accuracies.
The exact format of this file is now specified.
The file is a csv space delimited file, where each row contains 4 values:
1. An index of the optode (an integer numerical value representing the index of this optode. Starts from 0. any other value (strings, etc) will not be calibrated but still appear in the output file in the same standard coordiante system as the rest of the optodes).
2. 3 numerical values: X, Y, Z representing the location of this optode (note: values must be supplied in cm or inch).
The coordinate system these values are supplied at are not improtant, as they are transformed internally to a standard right-handed system.

## How to use GUI & Automatic calibration

The file [main.py](CapCalibrator/main.py) is the entry point of the application. In the most common use case, this script expects a path to a video file to be analyzed and a "template" model file discussed above.

`python main.py path_to_video_file path_to_template_file --mode manual`\
`python main.py path_to_video_file path_to_template_file --mode semi-auto`\
`python main.py path_to_video_file path_to_template_file --mode auto`

The mode "manual" indicates to the application that the user wants to annotate the video manually using the GUI. This is recommended for first time users so they can familiarize themselvse with the application.

The mode "semi-auto" indicates to the application that the user wants it to automatically annotate the video, but also to observe annotation results and to correct them if needed using the GUI. This is recommended when possible.

The mode "auto" indicates to the application that the user wants it to automatically annotate the video without any supervision. This is recommended for live sessions and when the system was oberved to perform well with a certain template model.


## Using a new cap (or how to improve accuracy)

After creating a template model for a new cap, we strongly recommend fine-tunning our supplied neural network for best results.
To do this, follow the steps below:

1. Create synthetic data using the [render script](DataSynth/render.py) (notice this script requires the template model file path and the renderer executable path as input). We recommend usings a minimum of 30000 iterations:\
   `python render.py path_to_template_model_file path_to_output_folder --exe path_to_renderer_executable --log path_to_output_log --iterations 30000`
2. Train the network on the images using the [train script](CapCalibrator/train.py)\
   When training is done, a model file will be availble in the [models](CapCalibrator/models) directory.
3. Use this model in the arguments supplied to the GUI / automatic calibration tools.

## List of mandatory optodes in template file

The following optodes must exist in the template model file (and their locations should corropond with real 10-20 locations as much as possible):
- "cz" : used to find standard coordinate and by renderer.
- "righteye" : used to find standard coordinate and by renderer.
- "lefteye" : used to find standard coordinate and by renderer.
- "nosetip" : used by renderer.
- "fp1" : used to find standard coordinate and by renderer.
- "fpz" : used to find standard coordinate and by renderer. **NOTE**: we actually used a location 1 cm above fpz in our experiments (can be seen marked by a green sticker in the example video) - This was shown to yield better results (but it is not mandatory).
- "fp2" : used to find standard coordinate and by renderer.

Use these names (exactly, without double quoutes) for the first field for them to parsed correctly.

## The standard coordiante system

This application outputs data in the following coordiante system (notice it is right-handed):
- x axis is from left to right ear
- y axis is from back to front of head
- z axis is from bottom to top of head
- the origin is defined by (x,y,z) = ((lefteye.x+righteye.x) / 2, cz.y, (lefteye.z+righteye.z) / 2)
- scale is cm. If "CZ" is too close to origin in terms of cm, this function scales it to cm (assuming it is inch)
