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


class Merge:
    def __call__(self, vertex1, vertex2, example_pairs):
        """
        Return the feature representations of the example pairs

        :param vertex1: Feature representation of the ligand residue
        :param vertex2: Feature representation of the receptor residue
        :param example_pairs: The example pairs we want to classify
        :return: a tensor of shape (number of examples in example_pairs * 2, vertex_feature_dimensions * 2)
        """
        # collect the features for the ligand protein residue
        out1 = tf.gather(vertex1, example_pairs[:, 0])

        # collect the features for the receptor protein residue
        out2 = tf.gather(vertex2, example_pairs[:, 1])

        # we will present to the model the representation of an example pair in both possible orders ie
        # (ligand_i, receptor_i) and (receptor_i, ligand_i)
        # We do this because the role of ligand/receptor is arbitrary, so we would like to
        # learn a scoring function that is independent of the order in which the two residues are presented to
        # the network.

        # a list of representations starting from ligands then receptors
        lr_order = tf.concat([out1, out2], axis=0)
        # a lilst of representations starting from receptors then ligands
        rl_order = tf.concat([out2, out1], axis=0)

        # stack the list sideways to get the paired representations for both (ligand_i, receptor_i) and
        # (receptor_i, ligand_i) form
        example_pairs_representation = tf.concat([lr_order, rl_order], axis=1)

        return example_pairs_representation


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
