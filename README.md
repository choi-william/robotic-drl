# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Capstone Project II codebase
* v1

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

### Running the code ###

`python3 ddpg.py`

Runs a pendulum environment by default.