import tensorflow as tf
import math


# this is a noise adaptation layer
def _nal(X, initial_bias):
    input_dim = X.shape.as_list()[1]
    size = initial_bias.shape[0]

    W = tf.get_variable(
        'weights', [input_dim, size, size], tf.float32,
        tf.zeros_initializer()
    )

    b = tf.get_variable(
        'bias', [size, size], tf.float32,
        tf.constant_initializer(initial_bias)
    )

    # (i,j,k - indices of a three dimensional tensor, in that order)
    # x - input vector
    # y - hidden true labels
    # z - observed noisy labels

    # this is equivalent to 'size' matrix products
    result = tf.add(tf.tensordot(X, W, [[1], [0]]), b)

    result = tf.nn.softmax(result)  # p(z=k|y=j,x=i)

    # p(z=k|y=j,x=i) - probability that for input vector 'i'
    # with true label 'j' we observe label 'k'

    return tf.transpose(result, [2, 0, 1])  # p(z=i|y=k,x=j)


# this constructs a feedforward fully connected neural network
# input dimension == X.shape[1] == architecture[0]
# output dimension == architecture[-1]
def _mapping(X, architecture, dropout, is_training):

    # number of layers
    depth = len(architecture) - 1
    result = X

    for i in range(1, depth + 1):
        with tf.variable_scope('layer_' + str(i)):
            result = _dropout(result, dropout[i - 1], is_training)
            result = _fully_connected(result, architecture[i])
            result = _batch_norm(result, is_training)
            result = _nonlinearity(result)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, result)

    return result


def _nonlinearity(X):
    return tf.nn.relu(X, name='ReLU')


def _dropout(X, rate, is_training):
    keep_prob = tf.constant(
        1.0 - rate, tf.float32,
        [], 'keep_prob'
    )
    result = tf.cond(
        is_training,
        lambda: tf.nn.dropout(X, keep_prob),
        lambda: tf.identity(X),
        name='dropout'
    )
    return result


def _batch_norm(X, is_training):
    return tf.contrib.layers.batch_norm(
        X, is_training=is_training, center=True,
        scale=False, fused=True, scope='batch_norm'
    )


def _affine(X, size):
    input_dim = X.shape.as_list()[1]
    maxval = math.sqrt(2.0/input_dim)

    W = tf.get_variable(
        'weights', [input_dim, size], tf.float32,
        tf.random_uniform_initializer(-maxval, maxval)
    )

    b = tf.get_variable(
        'bias', [size], tf.float32,
        tf.zeros_initializer()
    )

    return tf.nn.bias_add(tf.matmul(X, W), b)


def _fully_connected(X, size):
    input_dim = X.shape.as_list()[1]
    maxval = math.sqrt(2.0/input_dim)

    W = tf.get_variable(
        'weights', [input_dim, size], tf.float32,
        tf.random_uniform_initializer(-maxval, maxval)
    )

    return tf.matmul(X, W)


def _add_weight_decay(weight_decay):

    weight_decay = tf.constant(
        weight_decay, tf.float32,
        [], 'weight_decay'
    )

    trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    weights = [v for v in trainable if 'weights' in v.name]

    for W in weights:
        l2_loss = tf.multiply(
            weight_decay, tf.nn.l2_loss(W), name='l2_loss'
        )
        tf.losses.add_loss(l2_loss)
