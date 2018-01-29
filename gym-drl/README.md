# gym-drl #
gym-drl provides 2D environments designed to model a membrane platform that we built to test deep reinforcement learning algorithms. The package extends the existing set of environments available on openai-gym and adopts the structure for environments used by openai-gym. pyBox2D was used for physics simulation.

## Getting Started with the Simulated Environments ##
To only run the simulated environments, all you need to do is to follow the steps outlined below. If you want to interface with the hardware platform, please follow the remainder of instructions provided in the main  README file.

### Installation Instructions ###
Please note that the following instructions are only complete for an OSX user. We intend to expand upon it shortly to include linux.
We used python3 for this project. If you don't have an updated version of python already, you can get it using the following command.
Linux: `sudo apt-get install python3.6`
Mac: `brew install python3`

Following python packages need to be installed: tensorflow, tflearn, openai-gym, pyBox2D, pyGame.
* As mentioned above these are only the packages required for the simulated environments.



## Installation OSX ##
pyBox2D requires a set of packages to be pre-installed.
```
brew install cmake boost boost-python sdl2 swig wget
```

Once the above packages were installed clone the pyBox2D repository and install the package.
```
git clone https://github.com/pybox2d/pybox2d
cd pybox2d/
python setup.py clean
python setup.py build
python setup.py install
```

PyGame is also required to render the examples provided through pyBox2D. This can be installed using pip.
```
pip install pygame
```

OpenAI Gym is also required.
```
pip install gym
```

It's also recommended that a python virtual environment (venv) is used.

To install gym-drl, clone the repository and then subsequently install it using pip
```
cd gym-drl
pip install -e .
```

# Environments

Basketball
Bouncing
Moving to target
Ordering
Stacking
Rotating

