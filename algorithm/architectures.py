import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d

def architecture_actor_v0(s_dim, a_dim, action_amp, action_mid):
    inputs = tflearn.input_data(shape=[None, s_dim])

    net = tflearn.fully_connected(inputs, 400)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)
    net = tflearn.fully_connected(net, 300)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)

    # Final layer weights are init to Uniform[-3e-3, 3e-3]
    w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
    out = tflearn.fully_connected(net, a_dim, activation='tanh', weights_init=w_init)

    # Scale output to -action_bound to action_bound
    scaled_out = tf.multiply(out, action_amp)
    scaled_out = tf.add(out, action_mid)

    return inputs, out, scaled_out


def architecture_critic_v0(s_dim, a_dim):
    action = tflearn.input_data(shape=[None, a_dim])

    inputs = tflearn.input_data(shape=[None, s_dim])
    net = tflearn.fully_connected(inputs, 400)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)

    # Add the action tensor in the 2nd hidden layer
    # Use two temp layers to get the corresponding weights and biases
    t1 = tflearn.fully_connected(net, 300)
    t2 = tflearn.fully_connected(action, 300)

    net = tflearn.activation(
        tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

    # linear layer connected to 1 output representing Q(s,a)
    # Weights are init to Uniform[-3e-3, 3e-3]
    w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
    out = tflearn.fully_connected(net, 1, weights_init=w_init)

    return inputs, action, out

def architecture_imitate(s_dim, a_dim):
    inputs = tflearn.input_data(shape=[None, s_dim])

    net = tflearn.fully_connected(inputs, 400)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)
    net = tflearn.fully_connected(net, 300)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)

    # Final layer weights are init to Uniform[-3e-3, 3e-3]
    w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
    out = tflearn.fully_connected(net, a_dim, activation='tanh', weights_init=w_init)

    # Scale output to -action_bound to action_bound
    scaled_out = tf.multiply(out, action_amp)
    scaled_out = tf.add(out, action_mid)

    return inputs, out, scaled_out
