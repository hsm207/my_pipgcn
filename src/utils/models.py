from functools import reduce, partial

from utils.layers import *
from utils.losses import Weighted_CrossEntropy


class NoConvolutionModel:
    def __init__(self, legs_output_dims, dropout_rate=0.5, neg_wt=0.1, learning_rate=0.1):
        # a list of layers for the legs
        # the weight matrix for the first leg layer has height 70 because the input i.e. ligand and receptor
        # vetex is a 70-d vector
        legs_wt_dims = list(zip([70] + legs_output_dims[:-1], legs_output_dims))
        self.leg_fns = [tf.make_template('leg{}_no_conv'.format(i), No_Conv(dim, dropout_rate)) for i, dim in
                        enumerate(legs_wt_dims, 1)]

        # layers after the legs
        self.merge = tf.make_template('merge', Merge())
        # the shape after a merge is (batch_size * 2, output_dim_before_merge * 2)
        self.dense1 = tf.make_template('dense1', Dense((legs_output_dims[-1] * 2, 512), dropout_rate, 'relu'))
        self.dense2 = tf.make_template('dense2', Dense((512, 1), dropout_rate, 'linear'))
        self.avg_pred = tf.make_template('average_predictions', Average_Predictions())

        # loss
        self.loss_fn = tf.make_template('weighted_crossentropy', Weighted_CrossEntropy(neg_wt))
        # optimizer
        self.opt = tf.train.GradientDescentOptimizer(learning_rate)

    def __call__(self, vertex1, vertex2, examples):
        # since the vertex1 and vertex2 inputs are repeated batch_size times, we can just pass the first element
        # to extract their feature representations

        with tf.variable_scope('ligand_features'):
            feat_vertex1 = reduce(lambda f1, f2: f2(f1),
                                  self.leg_fns[1:],
                                  self.leg_fns[0](vertex1[0]))

        with tf.variable_scope('receptor_features'):
            feat_vertex2 = reduce(lambda f1, f2: f2(f1),
                                  self.leg_fns[1:],
                                  self.leg_fns[0](vertex2[0]))

        examples_representation = self.merge(feat_vertex1, feat_vertex2, examples)

        dense1 = self.dense1(examples_representation)
        dense2 = self.dense2(dense1)

        avg_preds = self.avg_pred(dense2)

        return avg_preds

    def compute_loss(self, labels, logits):
        loss = self.loss_fn(labels, logits)
        return loss

    def get_train_op(self, loss):
        # TODO: Figure out what is causing  UserWarning: Converting sparse IndexedSlices to a dense Tensor of
        # unknown shape. This may consume a large amount of memory "Converting sparse IndexedSlices to a dense
        # Tensor of unknown shape. "
        # See https://stackoverflow.com/questions/35892412/tensorflow-dense-gradient-explanation for potential
        # solutions
        train_op = self.opt.minimize(loss, tf.train.get_or_create_global_step())
        return train_op


# TODO: Rewrite these classes so that the common functions are reusable
class NodeAverageModel:
    def __init__(self, legs_output_dims, dropout_rate=0.5, neg_wt=0.1, learning_rate=0.1):
        # a list of layers for the legs
        # the weight matrix for the first leg layer has height 70 because the input i.e. ligand and receptor
        # vetex is a 70-d vector
        legs_wt_dims = list(zip([70] + legs_output_dims[:-1], legs_output_dims))
        self.leg_fns = [tf.make_template('leg{}_node_avg'.format(i), Node_Avg(dim, dropout_rate)) for i, dim in
                        enumerate(legs_wt_dims, 1)]

        # layers after the legs
        self.merge = tf.make_template('merge', Merge())
        # the shape after a merge is (batch_size * 2, output_dim_before_merge * 2)
        self.dense1 = tf.make_template('dense1', Dense((legs_output_dims[-1] * 2, 512), dropout_rate, 'relu'))
        self.dense2 = tf.make_template('dense2', Dense((512, 1), dropout_rate, 'linear'))
        self.avg_pred = tf.make_template('average_predictions', Average_Predictions())

        # loss
        self.loss_fn = tf.make_template('weighted_crossentropy', Weighted_CrossEntropy(neg_wt))
        # optimizer
        self.opt = tf.train.GradientDescentOptimizer(learning_rate)

    def __call__(self, features):
        # since the vertex1 and vertex2 inputs (and all other protein features) are repeated batch_size times, we
        # can just pass the first element to extract their feature representations

        l_vertex = features['l_vertex']
        r_vertex = features['r_vertex']
        paired_examples = features['example']
        l_nh_indices = features['l_hood_indices']
        r_nh_indices = features['r_hood_indices']

        # make the leg functions for this model a single argument function
        l_leg_fns = [partial(fn, nh_indices=l_nh_indices[0]) for fn in self.leg_fns]
        r_leg_fns = [partial(fn, nh_indices=r_nh_indices[0]) for fn in self.leg_fns]

        with tf.variable_scope('ligand_features'):
            l_feat_vertex = reduce(lambda f1, f2: f2(f1),
                                   l_leg_fns[1:],
                                   l_leg_fns[0](l_vertex[0]))

        with tf.variable_scope('receptor_features'):
            r_feat_vertex = reduce(lambda f1, f2: f2(f1),
                                   r_leg_fns[1:],
                                   r_leg_fns[0](r_vertex[0]))

        examples_representation = self.merge(l_feat_vertex, r_feat_vertex, paired_examples)

        dense1 = self.dense1(examples_representation)
        dense2 = self.dense2(dense1)

        avg_preds = self.avg_pred(dense2)

        return avg_preds

    def compute_loss(self, labels, logits):
        loss = self.loss_fn(labels, logits)
        return loss

    def get_train_op(self, loss):
        # TODO: Figure out what is causing  UserWarning: Converting sparse IndexedSlices to a dense Tensor of
        # unknown shape. This may consume a large amount of memory "Converting sparse IndexedSlices to a dense
        # Tensor of unknown shape. "
        # See https://stackoverflow.com/questions/35892412/tensorflow-dense-gradient-explanation for potential
        # solutions
        train_op = self.opt.minimize(loss, tf.train.get_or_create_global_step())
        return train_op


class NodeAverageEdgeModel:
    def __init__(self, legs_output_dims, dropout_rate=0.5, neg_wt=0.1, learning_rate=0.1):
        # a list of layers for the legs
        # the weight matrix for the first leg layer has height 70 because the input i.e. ligand and receptor
        # vetex is a 70-d vector
        legs_wt_dims = list(zip([70] + legs_output_dims[:-1], legs_output_dims))
        self.leg_fns = [tf.make_template('leg{}_node_and_edge_avg'.format(i), Node_Edge_Avg(dim, dropout_rate))
                        for i, dim in enumerate(legs_wt_dims, 1)]

        # layers after the legs
        self.merge = tf.make_template('merge', Merge())
        # the shape after a merge is (batch_size * 2, output_dim_before_merge * 2)
        self.dense1 = tf.make_template('dense1', Dense((legs_output_dims[-1] * 2, 512), dropout_rate, 'relu'))
        self.dense2 = tf.make_template('dense2', Dense((512, 1), dropout_rate, 'linear'))
        self.avg_pred = tf.make_template('average_predictions', Average_Predictions())

        # loss
        self.loss_fn = tf.make_template('weighted_crossentropy', Weighted_CrossEntropy(neg_wt))
        # optimizer
        self.opt = tf.train.GradientDescentOptimizer(learning_rate)

    def __call__(self, features):
        # since the vertex1 and vertex2 inputs (and all other protein features) are repeated batch_size times, we
        # can just pass the first element to extract their feature representations

        l_vertex = features['l_vertex']
        r_vertex = features['r_vertex']

        paired_examples = features['example']

        l_nh_indices = features['l_hood_indices']
        r_nh_indices = features['r_hood_indices']

        l_edge = features['l_edge']
        r_edge = features['r_edge']

        # make the leg functions for this model a single argument function
        l_leg_fns = [partial(fn, edges=l_edge[0], nh_indices=l_nh_indices[0]) for fn in self.leg_fns]
        r_leg_fns = [partial(fn, edges=r_edge[0], nh_indices=r_nh_indices[0]) for fn in self.leg_fns]

        with tf.variable_scope('ligand_features'):
            l_feat_vertex = reduce(lambda f1, f2: f2(f1),
                                   l_leg_fns[1:],
                                   l_leg_fns[0](l_vertex[0]))

        with tf.variable_scope('receptor_features'):
            r_feat_vertex = reduce(lambda f1, f2: f2(f1),
                                   r_leg_fns[1:],
                                   r_leg_fns[0](r_vertex[0]))

        examples_representation = self.merge(l_feat_vertex, r_feat_vertex, paired_examples)

        dense1 = self.dense1(examples_representation)
        dense2 = self.dense2(dense1)

        avg_preds = self.avg_pred(dense2)

        return avg_preds

    def compute_loss(self, labels, logits):
        loss = self.loss_fn(labels, logits)
        return loss

    def get_train_op(self, loss):
        # TODO: Figure out what is causing  UserWarning: Converting sparse IndexedSlices to a dense Tensor of
        # unknown shape. This may consume a large amount of memory "Converting sparse IndexedSlices to a dense
        # Tensor of unknown shape. "
        # See https://stackoverflow.com/questions/35892412/tensorflow-dense-gradient-explanation for potential
        # solutions
        train_op = self.opt.minimize(loss, tf.train.get_or_create_global_step())
        return train_op


class OrderDependentModel:
    def __init__(self, legs_output_dims, dropout_rate=0.5, neg_wt=0.1, learning_rate=0.1):
        # a list of layers for the legs
        # the weight matrix for the first leg layer has height 70 because the input i.e. ligand and receptor
        # vetex is a 70-d vector
        legs_wt_dims = list(zip([70] + legs_output_dims[:-1], legs_output_dims))
        self.leg_fns = [tf.make_template('leg{}_order_dependent'.format(i), Order_Dependent(dim, dropout_rate))
                        for i, dim in enumerate(legs_wt_dims, 1)]

        # layers after the legs
        self.merge = tf.make_template('merge', Merge())
        # the shape after a merge is (batch_size * 2, output_dim_before_merge * 2)
        self.dense1 = tf.make_template('dense1', Dense((legs_output_dims[-1] * 2, 512), dropout_rate, 'relu'))
        self.dense2 = tf.make_template('dense2', Dense((512, 1), dropout_rate, 'linear'))
        self.avg_pred = tf.make_template('average_predictions', Average_Predictions())

        # loss
        self.loss_fn = tf.make_template('weighted_crossentropy', Weighted_CrossEntropy(neg_wt))
        # optimizer
        self.opt = tf.train.GradientDescentOptimizer(learning_rate)

    def __call__(self, features):
        # since the vertex1 and vertex2 inputs (and all other protein features) are repeated batch_size times, we
        # can just pass the first element to extract their feature representations

        l_vertex = features['l_vertex']
        r_vertex = features['r_vertex']

        paired_examples = features['example']

        l_nh_indices = features['l_hood_indices']
        r_nh_indices = features['r_hood_indices']

        l_edge = features['l_edge']
        r_edge = features['r_edge']

        # make the leg functions for this model a single argument function
        l_leg_fns = [partial(fn, edges=l_edge[0], nh_indices=l_nh_indices[0]) for fn in self.leg_fns]
        r_leg_fns = [partial(fn, edges=r_edge[0], nh_indices=r_nh_indices[0]) for fn in self.leg_fns]

        with tf.variable_scope('ligand_features'):
            l_feat_vertex = reduce(lambda f1, f2: f2(f1),
                                   l_leg_fns[1:],
                                   l_leg_fns[0](l_vertex[0]))

        with tf.variable_scope('receptor_features'):
            r_feat_vertex = reduce(lambda f1, f2: f2(f1),
                                   r_leg_fns[1:],
                                   r_leg_fns[0](r_vertex[0]))

        examples_representation = self.merge(l_feat_vertex, r_feat_vertex, paired_examples)

        dense1 = self.dense1(examples_representation)
        dense2 = self.dense2(dense1)

        avg_preds = self.avg_pred(dense2)

        return avg_preds

    def compute_loss(self, labels, logits):
        loss = self.loss_fn(labels, logits)
        return loss

    def get_train_op(self, loss):
        # TODO: Figure out what is causing  UserWarning: Converting sparse IndexedSlices to a dense Tensor of
        # unknown shape. This may consume a large amount of memory "Converting sparse IndexedSlices to a dense
        # Tensor of unknown shape. "
        # See https://stackoverflow.com/questions/35892412/tensorflow-dense-gradient-explanation for potential
        # solutions
        train_op = self.opt.minimize(loss, tf.train.get_or_create_global_step())
        return train_op
