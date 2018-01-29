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


class Node_Avg:

    def __init__(self, weight_matrix_dim, dropout_rate):
        """
        Implements the convolution operator described by Equation 1 in the paper

        :param weight_matrix_dim: a 2-D tuple representing the dimension of the weight matrix
        :param dropout_rate: dropout rate for both dropout layers
        """
        self.dropout = Dropout(dropout_rate)
        self.center_weights = Dense_(units=weight_matrix_dim[1],
                                     activation='linear',
                                     use_bias=False,
                                     kernel_initializer=lambda *args, **kwargs: initializer('he', weight_matrix_dim),
                                     dtype=tf.float64,
                                     name='Wc')
        self.nh_weights = Dense_(units=weight_matrix_dim[1],
                                 activation='linear',
                                 use_bias=False,
                                 kernel_initializer=lambda *args, **kwargs: initializer('he', weight_matrix_dim),
                                 dtype=tf.float64,
                                 name='Wn')

        self.b = lambda: tf.get_variable(name='b',
                                         initializer=tf.zeros(weight_matrix_dim[1], dtype=tf.float64),
                                         dtype=tf.float64)

        self.relu = Activation('relu', name='ReLu')

    def __call__(self, vertices, nh_indices):
        # the indices of the neighbours of each residue in vertex
        nh_indices = tf.squeeze(nh_indices, axis=2)

        # count the number of neighbours for each residue in vertex
        # TODO: Figure out why need to add 1
        nh_sizes = tf.count_nonzero(nh_indices + 1, axis=1, dtype=tf.float64, keep_dims=True)

        # the convolution operator's center node term
        Zc = self.center_weights(vertices)

        # to calculate the convolution the convolution operator's neighbourhood term, we multiply all residues
        # in the vertices (each row) with the neighbourhood weight matrix. Then, using nh_indices, we take only
        # the values that are neighbourhood of a given residue
        v_Wn = self.nh_weights(vertices)

        Zn = tf.gather(v_Wn, nh_indices)  # (n_vertices, n_neighbours, n_v_Wn_columns)

        # now we element-wise sum all the neighbours for a given vertex
        Zn = tf.reduce_sum(Zn, axis=1)

        # divide each vertex feature with the number of neighbours
        Zn = tf.divide(Zn, tf.maximum(nh_sizes, tf.ones_like(nh_sizes, dtype=tf.float64)))

        sig = Zc + Zn + self.b()

        Z = self.relu(sig)

        Z = self.dropout(Z)

        return Z


class Node_Edge_Avg:
    def __init__(self, weight_matrix_dim, dropout_rate):
        self.dropout = Dropout(dropout_rate)
        self.center_weights = Dense_(units=weight_matrix_dim[1],
                                     activation='linear',
                                     use_bias=False,
                                     kernel_initializer=lambda *args, **kwargs: initializer('he', weight_matrix_dim),
                                     dtype=tf.float64,
                                     name='Wc')
        self.nh_weights = Dense_(units=weight_matrix_dim[1],
                                 activation='linear',
                                 use_bias=False,
                                 kernel_initializer=lambda *args, **kwargs: initializer('he', weight_matrix_dim),
                                 dtype=tf.float64,
                                 name='Wn')

        # the edges have 2 features only
        self.edge_weights = lambda: tf.get_variable(name='We',
                                                    initializer=initializer('he', (2, weight_matrix_dim[1])),
                                                    dtype=tf.float64)

        self.b = lambda: tf.get_variable(name='b',
                                         initializer=tf.zeros(weight_matrix_dim[1], dtype=tf.float64),
                                         dtype=tf.float64)

        self.relu = Activation('relu', name='ReLu')

    def __call__(self, vertices, edges, nh_indices):
        # the convolution operator's neighbourhood term. See Node_Average's __call__ method for details

        nh_indices = tf.squeeze(nh_indices, axis=2)
        nh_sizes = tf.count_nonzero(nh_indices + 1, axis=1, dtype=tf.float64, keep_dims=True)

        v_Wn = self.nh_weights(vertices)
        Zn = tf.gather(v_Wn, nh_indices)
        Zn = tf.reduce_sum(Zn, axis=1)
        Zn = tf.divide(Zn, tf.maximum(nh_sizes, tf.ones_like(nh_sizes, dtype=tf.float64)))

        # the convolution operator's edges term
        Ze = self._compute_edges_term(edges, nh_sizes)

        # the convolution operator's center node term
        Zc = self.center_weights(vertices)

        sig = Zc + Zn + Ze + self.b()

        Z = self.relu(sig)

        Z = self.dropout(Z)

        return Z

    def _compute_edges_term(self, edges, nh_sizes):
        # the shape of edges is (number of vertices, number of neighbours, 2)
        # the shape of the edge weights is (2, number of filters)
        # it is easier to compute the convolution operator on the edges using tensordot rather than using
        # Dense layers

        # the convolution operator on each feature on each neighbour for all vertices
        e_We = tf.tensordot(edges, self.edge_weights(), axes=[[2], [0]])

        # do an element wise sum over all the neighbours for each vertex
        Ze = tf.reduce_sum(e_We, axis=1)

        # divide each vertex in Ze by its number of neighbours
        Ze = tf.divide(Ze, tf.maximum(nh_sizes, tf.ones_like(nh_sizes, dtype=tf.float64)))

        return Ze


class Order_Dependent:
    def __init__(self, weight_matrix_dim, dropout_rate):
        self.dropout = Dropout(dropout_rate)
        self.center_weights = Dense_(units=weight_matrix_dim[1],
                                     activation='linear',
                                     use_bias=False,
                                     kernel_initializer=lambda *args, **kwargs: initializer('he', weight_matrix_dim),
                                     dtype=tf.float64,
                                     name='Wc')

        self.nh_weights = self._build_nh_weights(weight_matrix_dim)

        self.edge_weights = self._build_edge_weights(weight_matrix_dim[1])

        self.b = lambda: tf.get_variable(name='b',
                                         initializer=tf.zeros(weight_matrix_dim[1], dtype=tf.float64),
                                         dtype=tf.float64)

        self.relu = Activation('relu', name='ReLu')

    def __call__(self, vertices, edges, nh_indices):
        nh_indices = tf.squeeze(nh_indices, axis=2)
        nh_size = nh_indices.get_shape()[1].value

        # the convolution operator's neighbourhood term. See Node_Average's __call__ method for details
        Wn = self.nh_weights(nh_size)

        # for neighbour weight i, multiply it by by neighbour i across all the vertices
        Zn = tf.add_n([wt(tf.gather(vertices, nh_indices[:, i])) for i, wt in enumerate(Wn)])
        Zn = tf.divide(Zn, nh_size)

        # the convolution operator's edges term
        Ze = self._compute_edges_term(edges, nh_size)

        # the convolution operator's center node term
        Zc = self.center_weights(vertices)

        sig = Zc + Zn + Ze + self.b()

        Z = self.relu(sig)

        Z = self.dropout(Z)

        return Z

    def _build_nh_weights(self, weight_matrix_dim):
        """
        Utility funciton to build the unique weights for each neighbour
        :param weight_matrix_dim: The dimension of the weight matrix to build
        :return: A single argument function that takes the number of weights to build and returns
                 a list of those weights
        """

        def nh_weights_builder(n_neighbours):
            """
            Returns a list of tensors representing the unique weights for each neighbour
            :param n_neighbours: number of tensors to create in the list
            :return: A list of tensors
            """
            nh_weights = [Dense_(units=weight_matrix_dim[1],
                                 activation='linear',
                                 use_bias=False,
                                 kernel_initializer=lambda *args, **kwargs: initializer('he', weight_matrix_dim),
                                 dtype=tf.float64,
                                 name='Wn_{}'.format(i + 1)) for i in range(n_neighbours)]

            return nh_weights

        return nh_weights_builder

    def _build_edge_weights(self, n_filters):
        """
        Utility function to build the unique weights for the edges.

        The number of unique weights are is the same as the number of neighbours for a given residue

        :param n_filters: number of filters i.e. columns for the weight matrices
        :return: A function that takes the number of neighbours as an argument and returns a tensor of shape
                 (n_neighbours, 2, n_filters)
        """

        def edge_weights_buildfer(n_neightbours):
            edge_weights = tf.get_variable(name='We',
                                           initializer=initializer('he', (n_neightbours, 2, n_filters)),
                                           dtype=tf.float64)
            return edge_weights

        return edge_weights_buildfer

    def _compute_edges_term(self, edges, nh_size):
        # the shape of edges is (number of vertices, number of neighbours, 2)
        # the shape of the edge weights is (neighbours, 2, number of filters)
        # it is easier to compute the convolution operator on the edges using tensordot rather than using
        # Dense layers

        # the convolution operator (with unique weights) on each feature on each neighbour for all vertices
        Ze = tf.tensordot(edges, self.edge_weights(nh_size), axes=[[2, 1], [0, 1]])

        # divide each vertex in Ze by its number of neighbours
        Ze = tf.divide(Ze, 20)

        return Ze


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
