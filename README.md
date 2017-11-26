# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Capstone Project II codebase
* v1

### Files ###

#### algorithm/train ####

Trains an actor neural network based on the specified simulation

#### algorithm/execute ####

Executes a given actor neural network model in the specified simulation

#### algorithm/imitate ####

Takes in state-action pairs to train an actor neural network.

#### algorithm/replay_buffer ####

A supporting file to implement a replay buffer

#### algorithm/architectures ####

A supporting file that specified neural network architectures so they can be dynamically loaded.

#### camera/capture ####

A simple program to capture images from a camera, and includes several flags:

- -u: Undistorts a single image and tracking the position of a yellow object
- -t: Performs object tracking and provides sliders to tune. 's' can be pressed to save the parameters
- -a: Launches tracking which can be used in the arena. Also uses various flags for custom parameter loading

#### camera/calibrate ####

A program which uses a set of 10+ images to obtain camera parameters used to remove lens distortion

#### other/ddpg_visual ####

An experimental training file that takes in a pixel input

### Installation Instructions ###

#### Python Dependencies ####

##### Python3 #####
Linux: `sudo apt-get install python3.6`

Mac: `brew install python3`

##### Tensorflow #####

Tensorflow is our machine learning library that will handle all the hard ML stuff.

https://www.tensorflow.org/install/

I recommend installing through native-pip

Make sure that all installation is with python3 (eg. use pip3 instead of pip always)

##### TFLearn #####

TFLearn is a wrapper on top of tensorflow that simplifies it even more.

`pip3 install tflearn`

##### OpenAI gym #####

OpenAI gym is our reinforcement learning environment testing framework.

`pip3 install gym`

##### OpenCV #####

Installation instructions for ubuntu can be found here:

https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/

For mac, should be similar...hopefully

#### Misc Dependencies ####
`brew install ffmpeg` - Necessary to show simulation visuals

### Running the code ###

`python3 ddpg.py --render-env`

Runs a pendulum environment by default.

`tensorboard --logdir=robotic-ai/results/tf_ddpg/`

Outputs useful plots in realtime. (run in separate terminal window)