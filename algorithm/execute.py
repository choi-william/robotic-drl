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

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_amp, action_mid):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_amp = action_amp
        self.action_mid = action_mid

        # Actor Network
        with tf.variable_scope('actor_model'):
            self.inputs, self.out, self.scaled_out = self.create_actor_network()

    def create_actor_network(self):
        arch = 'architecture_actor_v1'
        get_net_method = getattr(architectures, arch)
        inputs, out, scaled_out = get_net_method(self.s_dim, self.a_dim, self.action_amp, self.action_mid)
        return inputs, out, scaled_out

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def run_sim(sess, env, actor):

    summary_ops, summary_vars = build_summaries()
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            a = actor.predict(np.reshape(s, (1, actor.s_dim)))
            s2, r, terminal, info = env.step(a[0])
            s = s2

            ep_reward += r

            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(j))))
                break

def main(args):

    with tf.Session() as sess:

        env = gym.make(args['env'])

        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        action_amp = (env.action_space.high-env.action_space.low)/2
        action_mid = (env.action_space.high+env.action_space.low)/2

        actor = ActorNetwork(sess, state_dim, action_dim, action_amp, action_mid)

        sess.run(tf.global_variables_initializer())

        actor_model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="actor_model")
        saver = tf.train.Saver(actor_model_variables)
        saver.restore(sess, args['model_dir']+'/'+args['model_name'] + '/' + args['model_name'])

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        run_sim(sess,env,actor)

        if args['use_gym_monitor']:
            env.monitor.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='../results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='../results/tf_ddpg')
    parser.add_argument('--model-dir', help='directory for storing saved models', default='../results/models')
    parser.add_argument('--model-name', help='name of the saved model')

    parser.set_defaults(render_env=True)
    parser.set_defaults(use_gym_monitor=False)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
