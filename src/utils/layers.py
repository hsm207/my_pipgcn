import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Activation, Dropout


class No_Conv:
    """
    Implements a layer that does no convolution i.e. Equation 1 with no summation over neighbours

    This layer first does the following, given an input:

        1. Apply a dropout layer
        2. Perform the matrix multiplication on the weight matrix and add the bias terms
        3. Apply the ReLu activation function
        4. Apply a dropout layer AGAIN.

    :param weight_matrix_dim: a 2-D tuple representing the dimension of the weight matrix
    :param dropout_rate: dropout rate for both dropout layers
    """
    def __init__(self, weight_matrix_dim, dropout_rate):
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.weight_matrix_dim = weight_matrix_dim
        self.relu = Activation('relu', name='ReLu')

    def __call__(self, x):
        with tf.variable_scope('kernel'):
            weight_matrix = tf.get_variable('Wvc', dtype=tf.float64,
                                            initializer=initializer('he', self.weight_matrix_dim))
            bias_weights = tf.get_variable('bv', dtype=tf.float64,
                                           initializer=tf.zeros(self.weight_matrix_dim[1], dtype=tf.float64),
                                           )
        x = self.dropout1(x)
        x = tf.matmul(x, weight_matrix, name='Zc') + bias_weights
        x = self.relu(x)
        x = self.dropout2(x)

        return x


# taken directly from the original author's source
def initializer(init, shape):
    if init == "zero":

        return tf.zeros(shape)

    elif init == "he":
        # TODO: Figure out formula's derivation
        # The paper: https://arxiv.org/pdf/1502.01852.pdf
        fan_in = np.prod(shape)

        std = 1 / np.sqrt(fan_in)

        return tf.random_uniform(shape, minval=-std, maxval=std, dtype=tf.float64)
