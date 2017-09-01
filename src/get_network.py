import tensorflow as tf
from input_utils import _get_data
from parts_of_the_net import _mapping, _add_weight_decay, _nal, _nonlinearity, _affine, _dropout


def get_network(initial_bias, architecture, dropout, optimizer, weight_decay=None):
    """Create a computational graph of neural network with noise adaptation layer.

    Arguments:
        initial_bias: A numpy array of shape [n_classes, n_classes]
            and of type 'float32'. It is used for initialization of noise
            adaptation layer's biases.
        architecture: A list that contains number of hidden units for each layer,
            where architecture[0] equals to the number of input features,
            architecture[-1] equals to the number of classes.
        dropout: A list that contains dropout rate for each layer.
            It is required that len(dropout) == len(architecture) - 1.
        optimizer: A Tensorflow optimizer.
        weight_decay: A scalar or None.

    For example:
        architecture=[54, 100, 100, 7],
        dropout=[0.2, 0.5, 0.1]

    See openreview.net/forum?id=H12GRgcxg for a description of the noise adaptation layer.

    """

    graph = tf.Graph()
    with graph.as_default():

        with tf.variable_scope('control'):
            is_training = tf.placeholder_with_default(True, [], 'is_training')

        input_dim = architecture[0]
        num_classes = architecture[-1]

        with tf.device('/cpu:0'), tf.variable_scope('input_pipeline'):
            data_init, x_batch, y_batch = _get_data(num_classes, input_dim, is_training)

        with tf.variable_scope('inputs'):
            X = tf.placeholder_with_default(x_batch, [None, input_dim], 'X')
            Y = tf.placeholder_with_default(y_batch, [None, num_classes], 'Y')

        h = _mapping(X, architecture[:-1], dropout[:-1], is_training)
        h = _nonlinearity(h)
        h = _dropout(h, dropout[-1], is_training)

        with tf.variable_scope('softmax'):
            logits = _affine(h, num_classes)

            # x - input vector
            # y - hidden true labels
            # z - observed noisy labels

            predictions = tf.nn.softmax(logits)  # p(y=j|x=i)

            # p(y=j|x=i) - probability that input vector 'i'
            # has true label 'j'

        with tf.variable_scope('nal'):
            correction = _nal(h, initial_bias)  # p(z=i|y=k,x=j)

        with tf.variable_scope('correction'):

            # p(y=k|x=j)*p(z=i|y=k,x=j)
            result = tf.multiply(predictions, correction)

            # p(z=i|x=j) = sum_k p(y=k|x=j)*p(z=i|y=k,x=j)
            result = tf.reduce_sum(result, axis=2)

            # p(z=j|x=i)
            corrected_predictions = tf.transpose(result, [1, 0])

            # p(z=j|x=i) - probability that for input vector 'i'
            # we observe noisy label 'j'

        with tf.variable_scope('train_log_loss'):
            # this is not true logloss because we assume that input
            # labels Y are noisy
            train_log_loss = tf.losses.log_loss(Y, corrected_predictions)

        with tf.variable_scope('total_loss'):
            total_loss = tf.losses.get_total_loss()

        if weight_decay is not None:
            with tf.variable_scope('weight_decay'):
                _add_weight_decay(weight_decay)

        trainable = tf.trainable_variables()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops), tf.variable_scope('optimizers'):
            # we train the model in two stages:
            # in the the first stage, weights of NAL are fixed
            # but NAL's biases are not
            grads_and_vars1 = optimizer.compute_gradients(
                total_loss, var_list=[v for v in trainable if v.name != 'nal/weights:0']
            )
            optimize1 = optimizer.apply_gradients(grads_and_vars1)

            # in the second stage we use all parameters
            grads_and_vars2 = optimizer.compute_gradients(total_loss)
            optimize2 = optimizer.apply_gradients(grads_and_vars2)

        with tf.variable_scope('utilities'):
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            is_equal = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))
            log_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
            )

        summaries = _add_summaries()

    graph.finalize()
    ops = [
        data_init, predictions, log_loss, train_log_loss,
        optimize1, optimize2, init, saver, accuracy, summaries
    ]
    return graph, ops


def _add_summaries():
    # add histograms of all trainable variables and of all layer activations

    summaries = []
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    activations = tf.get_collection(tf.GraphKeys.ACTIVATIONS)

    for v in trainable_vars:
        summaries += [tf.summary.histogram(v.name[:-2] + '_hist', v)]
    for a in activations:
        summaries += [tf.summary.histogram(a.name[:-2] + '_activ_hist', a)]

    return tf.summary.merge(summaries)
