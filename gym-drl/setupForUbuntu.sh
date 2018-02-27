#!/bin/bash



sudo apt-get install python3.6
pip3 install --user tensorflow
pip3 install --user tflearn
pip3 install --user gym
sudo apt-get install cmake sdlbasic swig wget
git clone https://github.com/pybox2d/pybox2d
pushd pybox2d/
	python3 setup.py clean
	python3 setup.py build
	python3 setup.py install --user
popd
pip3 install pygame
pip3 install --user -e .
