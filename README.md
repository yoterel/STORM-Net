# STORM: Simple and Timely Optode Registration Method for Functional Near-Infrared Spectroscopy (FNIRS)
All files in the repository are an implementation of the original manuscript and research which is not yet published.
## Introduction
This application is designed to provide an accurate estimation of the position of an fNIRS probing cap on a participant’s head, based on a short video. In other words, given a short video of the participant wearing the fNIRS cap, the application outputs the coordiantes of every optode position in MNI coordinates (or some other coordiante system that can be easly transformed to MNI using external tools (such as [SPM fNIRS](https://www.nitrc.org/projects/spm_fnirs/)).

It contains three seperate tools:
1. A GUI that allows supervising the automatic method ("semi-supervised") in a controlled manner - this is recommended for first time users.
2. A ready-to-use end-to-end script that performs calibration given a video file - use this when you are comfortable with the GUI and its results.
3. A bundle of python scripts for finetunning (or training from scratch) the neural networks we provide - this is suggested for advanced users who wish to customize or improve some block in the pipeline. Note this also includes a synthetic data generator implemented in [Unity](https://unity.com/).

### Application dependencies:
-	[x] Python 3.6 or higher (required by (1), (2), and (3))
-	[x] Use the requirments.txt file to obtain all python library dependencies (we suggest using a seperate environment such as conda or virtualenv):\
      `pip install -r requirements.txt`\
-     [x] Neural-network model files which can be downloaded from [here](https://www.cs.tau.ac.il/~yotamerel/models/storm_models.zip).\
      Place the files under CapCalibrator/models (after extracting)
-	[x] Unity 2019.3 or higher (required by (3) if changes are made to the synthetic data renderer)


### The template model file

The application uses a pre-measured template model file as an input. An example of such file is located under the [example_models](example_models) directory.
This template model was obtained using a 3D digitizer, but can similarly be obtained in any way as long as the measurements are accurate to a satisfactory degree.
It is strongly recommended to create your own template model file **per physical cap model you are using in your lab**, this will lead to best accuracies.
The exact format of this file is now specified.
The file is a csv space delimited file, where each row contains 4 values:
1. A "name" of the optode (some identifier that distinguishes it from the rest, it can also be a number)
2. 3 numerical values: X, Y, Z representing the location of this optode (note: values must be supplied in cm or inch).
The coordinate system these values are supplied at are not improtant, as they are transformed internally to a standard right-handed system.

### How to use GUI & Automatic calibration

The file main.py is the entry point of the application. In the most common use case, this script expects a path to a video file to be analyzed. Notice another mandatory input is a "template" model file, which is a text file containing data of the locations of all relevant optodes in some arbitrary global coordinate system (further details below).

`python main.py path_to_video_file path_to_template_file --mode manual`\
`python main.py path_to_video_file path_to_template_file --mode semi-auto`\
`python main.py path_to_video_file path_to_template_file --mode auto`\

The mode "manual" indicates to the application that the user wants to annotate the video manually using the GUI. This is recommended for first time users so they can familiarize themselvse with the application.

The mode "semi-auto" indicates to the application that the user wants it to automatically annotate the video, but also to observe annotation results and to correct them if needed using the GUI. This is recommended when possible.

The mode "auto" indicates to the application that the user wants it to automatically annotate the video without any supervision. This is recommended for live sessions and when the system was oberved to perform well with a certain template model.


### How to improve accuracy - strongly recommended when using a new template model

After creating a template model, we strongly recommend fine-tunning our supplied neural network.
To do this, follow the steps below:

1. Create synthetic data using the [render script](DataSynth/render.py) (this python script requires the template model as input).\
   We recommend usings a minimum of 30000 iterations and the --transform argument. Notice this script requires an executable path to the renderer.
2. Train the network on the images using the [train script](CapCalibrator/train.py)\
   When training is done, a model file will be availble in the [models](CapCalibrator/models) directory .
3. Use this model in the arguments supplied to the GUI / automatic calibration tools.
