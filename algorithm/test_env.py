import tensorflow as tf
import numpy as np
import gym
import gymdrl
from gym import wrappers
import gymdrl
import tflearn
import argparse
import pprint as pp
import architectures

# USAGE
# python3 execute.py --render-env --model-name membrane_test2

def main(args):

    env = gym.make(args['env'])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    action_amp = (env.action_space.high-env.action_space.low)/2
    action_mid = (env.action_space.high+env.action_space.low)/2


    while True:
        s = env.reset()
        for j in range(200):
            env.render()
            a = np.random.rand(5)*2-1
            a = np.clip(a, env.action_space.low, env.action_space.high)
            env.step(a)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    
    args = vars(parser.parse_args())
    
    main(args)
