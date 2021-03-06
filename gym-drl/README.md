# gym-drl #
gym-drl provides 2D environments designed to model a membrane platform that we built to test deep reinforcement learning algorithms.  
The package extends the existing set of environments available on openai-gym and adopts the structure for environments used by openai-gym.  
pyBox2D was used for physics simulation.

## Getting Started with the Simulated Environments ##
To only run the simulated environments, all you need to do is to follow the steps outlined below.  
If you want to interface with the hardware platform, please follow the remainder of instructions provided in the main  README file.  
It's also recommended that a python virtual environment (venv) is used.

### Installation Instructions ###
Please note that the following instructions are only complete for an OSX user. We intend to expand upon it shortly to include linux.  
We used python3 for this project. If you don't have an updated version of python already, you can get it using the following command.  
Linux: `sudo apt-get install python3.6`  
Mac: `brew install python3`  

Following python packages need to be installed: _tensorflow, tflearn, openai-gym, pyBox2D, pyGame_.  
As mentioned above these are only the packages required for the simulated environments.  

##### Tensorflow #####
Tensorflow is our machine learning library that will handle all the hard ML stuff: https://www.tensorflow.org/install/  
I recommend installing through native-pip. Make sure that all installation is with python3 (eg. use pip3 instead of pip always)  
```pip3 install tensorflow```

##### Tflearn #####
TFLearn is a wrapper on top of tensorflow that simplifies it even more.  
```pip3 install tflearn```

##### Openai-gym #####
OpenAI gym is our reinforcement learning environment testing framework.  
```pip3 install gym```

##### PyBox2D #####
pyBox2D requires a set of packages to be pre-installed.  
```brew install cmake boost boost-python sdl2 swig wget```
```sudo apt-get install cmake sdlbasic swig wget```

Once the above packages were installed clone the pyBox2D repository and install the package.  
```
git clone https://github.com/pybox2d/pybox2d
cd pybox2d/
python3 setup.py clean
python3 setup.py build
python3 setup.py install --user
```

##### PyGame #####
PyGame is required to render the examples provided through pyBox2D. This can also be installed using pip.  
```pip3 install pygame```

##### FFmpeg #####
Necessary to show simulation visuals  
```brew install ffmpeg```

##### Finally installing gym-drl #####
To install gym-drl, clone the repository and then subsequently install it using pip
```
cd gym-drl
pip3 install --user -e .
```

### Running an Environment ###

There are several environments you can choose to train on. More details regarding the environments are provided below in the _Environments_ section.  

##### Training #####
To start training you have to execute _train.py_ located within the _algorithm_ folder as shown below:  
```python3 train.py --env <environment name> --output-name <output folder name>```  

Running a training session for the first time creates a _results_ folder at the top level of this repository.  
This folder contains two subfolders: _models and tf_ddpg_.  
The _models_ folder contains the learned model following training, which could be used for transfer learning  
and the _tf_ddpg_ folder contains the statistics (reward, qmax) during the training that could be visualized using tensorboard.  

Following each training session, a folder by the "output folder name" that you specified when you ran the train command will be created  
in both the _models_ and _tf_ddpg_ folders containing all the relevant data regarding training. So be careful not to use the same output folder name  
for different training sessions. This will result in the data being overwritten.  

To visualize the data stored in the _tf_ddpg_ folder run the following command:  
```tensorboard --logdir results/tf_ddpg/<output folder name>/. ```  
This outputs useful plots in realtime. We recommend running this in a separate terminal window.  

Note: In this case where one desires a policy to be trained from another policy (to continue training, or attempt transfer learning) the _--input-name_ flag can be used.

```python3 train.py --env <environment name> --output-name <output folder name> --input-name <input model name>```  
Where the input model name is a previously trained model that exists in results/models/

##### Executing #####
To start executing the model that you've trained, you have to execute _execute.py_ located wihtin the _algorithm_ folder as shown below:  
```python3 execute.py --env <environment name> --model-name <model folder name>```

The "model folder name" is the output folder name specified during the training session.

### Environments ###

##### Basketball #####
Shoots the ball through the hoop.  
```python3 train.py --env MembraneBasket-v0 --output-name <output folder name>```

##### Bouncing #####
Bounces the ball at the center of the platform at a specified height.  
```python3 train.py --env MembraneJump-v0 --output-name <output folder name>```

##### Moving to target #####
Moves the ball to a specified target position (x,y). The ball's starting position is arbitary.  
```python3 train.py --env MembraneTarget-v0 --output-name <output folder name>```

##### Ordering #####
Moves the green box to be before the yellow box. The box's are also horizontally aligned.  
```python3 train.py --env MembraneOrder-v0 --output-name <output folder name>```

##### Stacking #####
Stacks the green box on top of the yellow box.  
```python3 train.py --env MembraneStack-v0 --output-name <output folder name>```

##### Rotating #####
Rotates the square box by 180 degrees (angle could be adjusted).  
```python3 train.py --env MembraneRotate-v0 --output-name <output folder name>```

### Creating a custom environment ###

Creating a new environment can be accomplished by modifying the two __init__.py files in this repo as described in:  
https://github.com/openai/gym/tree/master/gym/envs

When creating an environment that can transfer the hardware, it is highly recommended to use the helper functions in gymdrl/envs/membrane_base, as done in the other environments in this repo.  
This performs a lot of the membrane generation, and leaves one to design a reward function, the state vector, and the objects to manipulate.  
In retrospect, this should have been implemented with class inheritance, but a temporary fix spun out of control. 

### Training on Hardware ###

Provided the following:  
  - All necessary hardware dependencies are installed (see root README)  
  - Arduino is plugged into the computer  
  - Arduino has the serial communication program flashed on it  
  - Webcam connected to computer  
  - Power supply is connected to the hardware (and changed to 7V)  

Then, one can run the hardware environment through gym in the regular ways described above:  
```python3 train.py --env MembraneHardware-v0 --output-name <output folder name>```  
```python3 execute.py --env MembraneHardware-v0 --model-name <model name>```  

The reward function of this environment is currently fixed but can be modified.  


