import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Activation, Dropout
from tensorflow.python.keras.layers import Dense as Dense_


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


class Dense:
    def __init__(self, weight_matrix_dim, dropout_rate=0.5, activation_fn='relu'):
        """
        Create a fully connected layer.

        Given an inpux x, this layer does the following:

            1. Applies droppout
            2. Applies a dense layer
            3. Applies dropout again

        The weights in the dense layer are initialized using He initialization and the biases are initialized to 0.

        :param weight_matrix_dim: The dimension of the dense layer (num_of_inputs, num_of_outputs)
        :param dropout_rate: The dropout layer for bot dropoout layers
        :param activation_fn: A string representing the type of activation function to use at the dense layer
        """
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dense = Dense_(units=weight_matrix_dim[1], activation=activation_fn, bias_initializer='zeros',
                            kernel_initializer=lambda *args, **kwargs: initializer('he', weight_matrix_dim),
                            dtype=tf.float64)

    def __call__(self, x):
        x = self.dropout1(x)
        x = self.dense(x)
        x = self.dropout2(x)

        return x


class Average_Predictions:
    def __call__(self, x):
        """
        This layer computes the average predictions given a batch of tensors containing predictions for the
        (ligand, receptor) and (receptor, ligand) pair.

        :param x: A tensor of shape (batch_size*2, 1). The first batch_size tensors are the predictions for
                  (ligand_i, receptor_i) pairs and the next batch_size of tensors are the predictions for the
                  (receptor_i, ligand_i) pairs.
        :return: A tensor of shape (batch_size, 1) representing the averaged predidctions of (ligand_i, receptor_i) and
                 (receptor_i, ligand_i)
        """
        # split the input back into 2 list of tensors, one with the form (ligand_i, receptor_i) and the other
        # (receptor_i, ligand_i)

        representations = tf.split(x, 2, axis=0)

        # compute the average of the prediction for (ligand_i, recepter_i) and (receptor_i, ligand_i), i.e.
        # same pair of ligand and receptor, just in different order
        mean_prediction = tf.reduce_mean(tf.stack(representations, axis=0), axis=0)
        return mean_prediction


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