"""
This module implements the ROC AUC metric that is used to evaluate the models.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve

import utils.misc


class AUC_Calculator:
    def __init__(self):
        self.sess = tf.Session()
        self.roc_auc = []

    def _get_labels(self, input_fn):
        """
        Extract the labels from an estimator's input function

        :param input_fn: An input function to an estimator
        :return: A numpy array containing all the labels in input_fn
        """
        labels = []

        # input_fn returns a tuple of dictionaries (features, labels).
        labels_iter = input_fn()[1]['label']

        while True:
            # iterate over the current input function until it is exhausted.
            # while iterating, store the labels
            try:
                l = self.sess.run(labels_iter)
                labels.append(l)

            except tf.errors.OutOfRangeError:
                # when the iterator is exhausted, concatenate the accumulated labels
                labels = np.concatenate(labels, axis=0)
                break

        return labels

    def compute_roc_auc(self, estimator, input_fns):
        """
        Compute the estimator's ROC AUC metric.

        The metric is the ROC AUC averaged over input_fns ie average ROC AUC per protein.

        :param estimator: The model to compute the metric on
        :param input_fns: A list of input functions for the model to do the predictions
        :return: A float representing the mean ROC AUC
        """
        for i, input_fn in enumerate(input_fns, 1):
            print('{}: Computing ROC AUC for input function {}...\n'.format(utils.misc.current_time(), i))
            predictions = estimator.predict(input_fn)
            # extract the prediction values from the estimator's predictions
            predictions = np.fromiter((obs['score'][0] for obs in predictions), dtype=np.float64)

            labels = self._get_labels(input_fn)

            fpr, tpr, _ = roc_curve(labels, predictions)
            roc_auc = auc(fpr, tpr)
            self.roc_auc.append(roc_auc)

        mean_roc_auc = np.mean(self.roc_auc)

        return mean_roc_auc
