import tensorflow as tf


class Weighted_CrossEntropy:
    def __init__(self, neg_wt):
        self.neg_wt = neg_wt

    def _ohe_labels(self, labels):
        """
        One hot encode a a batch of labels.

        :param labels: A tensor of shape (batch_size,) where each component is either -1 or 1
        :return: A tensor of shape (batch_size, 2) representing the one hot encoded values of labels.
                 The first column corresponds to label value -1, the second column corresponds to label value 1.
        """
        # labels can take on the value of -1 or 1
        # we will return a two-column tensor where the value of the first column is 1 if the label is -1 and 0
        # otherwise. The value of the second column is 1 if the label is 1 and 0 otherwise.

        # negative indices are mapped are not reflected in the output of tf.one_hot (all columns are 0) so we
        # need to add 1 to labels, call tf.one_hot with depth 3 and extract the first and last column only (since -1 is
        # now 0 and 1 is now 2).

        ohe_labels = tf.gather(tf.one_hot(labels + 1, 3),
                               [0, 2],
                               axis=1)

        return ohe_labels

    def _adapt_model_logits(self, logits):
        """
        Convert the logits (output of the model) into a form suitable to be passed to the softmax_cross_entropy loss
        function.

        :param logits: A tensor of shape (batch_size, 1) representing the output (average prediction) from the model
        :return: A tensor of shape (batch_size, 2) where the first column is the logit for label -1 and the second
                 column is the logit for label 1.
        """

        logits_for_softmax = tf.concat([-logits, logits], axis=1)

        return logits_for_softmax

    def __call__(self, labels, logits):
        # set the weights for the negative examples i.e -1 to the negative weight
        weights = self.neg_wt * ((labels - 1) / -2)
        # the weigths of the positive examples are now 0, so set it back to 1
        weights = weights + (labels + 1) / 2
        ohe_labels = self._ohe_labels(labels)
        expanded_logits = self._adapt_model_logits(logits)

        loss = tf.losses.softmax_cross_entropy(ohe_labels, expanded_logits, weights=weights)

        return loss