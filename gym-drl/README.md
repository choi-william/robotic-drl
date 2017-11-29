# gym-drl
gym-drl provides two 2D environments designed to model a platform that we built to test deep reinforcement learning algorithms. pyBox2D was used for physics simulation

# Installation OSX
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

# Example Code
```
import gym
import gymdrl
env = gym.make('')
env.reset()
env.render()
```