# DEEEP REINFORCEMENT LEARNING PLATFORM #

### What is this repository for? ###

* Capstone Project II codebase
* v1

### Installation Instructions ###

**If you intend to only run the simulation, all the information you need including the installation instructions  
is provided in the README.md file under the _gym-drl_ folder.  
i.e. robotic-ai/gym-drl/README.md** 

There are couple other dependencies to be resolved if you intend to run the hardware platform as well.  
If space is available, we suggest installing these as well after following the installation instructions in the above link.  

##### OpenCV #####
Only necessary for hardware interfacing. Installation instructions for ubuntu can be found here:  
https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/  

If you want to avoid using virtualenvironments (not that they're bad)  
```pip3 install opencv-python```

##### PySerial #####
Used for arduino communication. Only necessary for hardware interfacing. 
```pip3 install pyserial```

### File Structure ###

_robotic-ai/algorithm_ contains the ddpg algorithm and files associated with training  
_robotic-ai/gym-drl_ contains the hardware and simulated environments and files for interfacing with the hardware platform.  
##### algorithm/train #####

Trains an actor neural network based on the specified simulation. Details are found in gym-drl/README.md.

##### algorithm/execute #####

Executes a given actor neural network model in the specified simulation. Details are found in gym-drl/README.md.

##### algorithm/imitate #####

Takes in state-action pairs to train an actor neural network.

##### algorithm/replay_buffer #####

A supporting file to implement a replay buffer

##### algorithm/architectures #####

A supporting file that specified neural network architectures so they can be dynamically loaded.

##### camera/capture #####

A simple program to capture images from a camera, and includes several flags:

- -u: Undistorts a single image and tracking the position of a yellow object
- -t: Performs object tracking and provides sliders to tune. 's' can be pressed to save the parameters
- -a: Launches tracking which can be used in the arena. Also uses various flags for custom parameter loading

##### camera/calibrate #####

A program which uses a set of 10+ images to obtain camera parameters used to remove lens distortion

##### other/ddpg_visual #####

An experimental training file that takes in a pixel input

