""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp
import architectures

import os
import re

def load_data(data):

    #probably data will be json format, an array of objects with an observation subobject, and an action subobject

    #return array with features and labels.

def format_data(data_json):

    result_state = []
    result_action = []

    action_amp = []
    action_mid = []

    for i in range(len(data_json)):
        current_state = []
        current_action = []
        for key in data_json['data'][i]['state']:
            current_state = current_state + data_json[i]['data']['state'][key]
        for key in data_json[i]['action']
            current_action = current_action + data_json[i]['data']['action'][key]

            if i==1:
                action_amp = action_amp + data_json['meta']['action_amps'][key]
                action_mid = action_mid + data_json['meta']['action_mids'][key]

        result_state.append(current_state)
        result_action.append(current_action)

    return result_state, result_action, action_amp, action_mid

def get_meta(data_json):

def manage_model_dir(model_name,model_dir):
     #if no model name is set, create a unique one
    if model_name == 'unnamed':
        maxnum = -1
        for name in os.listdir(model_dir):
            if name.startswith('unnamed'):
                num = int(re.search(r'\d+$', name).group(0)) #find ending number of file name
                if num > maxnum:
                    maxnum = num
        model_name = 'unnamed' + str(maxnum+1)

    model_dir = model_dir+'/'+model_name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

def create_actor_network(state_size, action_size, action_amp, action_mid):
    arch = 'architecture_actor_v0'
    get_net_method = getattr(architectures, arch)
    inputs, out, scaled_out = get_net_method(state_size, action_size, action_amp, action_mid)    
    return inputs, out, scaled_out

def main(args):

    LEARNING_RATE = 0.0001

    manage_model_dir(args['model_name'],args['model_dir'])
    json = load_data(args['data_path'])
    states,actions,amp,mid = format_data(json)


    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        with tf.variable_scope('actor_model'):
            inputs, out, scaled_out = create_actor_network(size(state,2),size(action,2),amp,mid)

        # Saver
        actor_model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="actor_model")
        saver = tf.train.Saver(actor_model_variables)

        ground_truth_actions = tf.placeholder(tf.float32, [None, size(action,2)])

        # Define loss and optimization Op
        loss = tflearn.mean_square(ground_truth_actions, out)
        optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        sess.run([out, optimize], feed_dict={
            inputs: states,
            ground_truth_actions: actions
        })

        saver.save(sess, args['model_dir']+'/'+args['model_name'] + '/' + args['model_name'])

        #SAVE MODEL
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for training')

    # run parameters

    parser.add_argument('--data-path', help='path of data to train on',)
    parser.add_argument('--model-dir', help='directory for storing saved models', default='../results/models')
    parser.add_argument('--model-name', help='name of the saved model', default='unnamed')

    parser.set_defaults(load_model=False)

    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
