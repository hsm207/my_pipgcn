import gzip
import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import get_file


class NoDiffusionDataset:
    def __init__(self, batch_size=128, repeat=1):
        # TODO: Set separate argument for test and batch size
        self.dataset_dir = '../datasets'

        self.trainfile = 'train.cpkl'
        self.trainurl = 'https://zenodo.org/record/1127774/files/train.cpkl.gz'

        self.testfile = 'test.cpkl'
        self.testurl = 'https://zenodo.org/record/1127774/files/test.cpkl.gz'

        self.batch_size = batch_size
        self.repeat = repeat

    def _downnload_file(self, filename, fileurl):
        filepath = os.path.abspath(os.path.join(self.dataset_dir, filename))
        return get_file(filepath, fileurl, extract=True)

    def _parse_file(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            _, data = pickle.load(f, encoding='latin1')
        return data

    def _build_protein_dataset(self, protein_data, type='train'):
        def input_fn():
            # complex_code is not a feature used in the models
            protein_features = {k: v for k, v in protein_data.items() if k not in ('label', 'complex_code')}

            # protein features are constant for the whole paired examples
            paired_examples = protein_data['label']

            protein_feat_labels = protein_features.keys()
            protein_feat_values = [np.expand_dims(val, 0) for val in protein_features.values()]

            # repeat the protein features batch_size times
            protein_ds = tf.data.Dataset.from_tensor_slices(tuple(feat_val for feat_val in protein_feat_values)) \
                .map(lambda *vals: dict(zip(protein_feat_labels, vals))) \
                .repeat()

            # the first 2 columns in the 'label' key are part of the features used in the model
            # the 3rd column in the 'label' key is the target variable
            paired_ds = tf.data.Dataset.from_tensor_slices(paired_examples) \
                .map(lambda row: (row[:2], {'label': row[2]}))

            ds = tf.data.Dataset.zip((protein_ds, paired_ds))

            if type == 'train':
                ds = ds.shuffle(buffer_size=100000)

            ds = ds \
                .batch(self.batch_size) \
                .repeat(self.repeat)

            features, (example, label) = ds.make_one_shot_iterator().get_next()

            features['example'] = example

            return features, label

        return input_fn

    def _get_input_fn(self, type):
        if type == 'train':

            file = self._downnload_file(self.trainfile, self.trainurl)

        else:
            file = self._downnload_file(self.testfile, self.testurl)

        data = self._parse_file(file)

        input_fns = [self._build_protein_dataset(protein, type) for protein in data]

        return input_fns

    def get_input_fns(self):
        """
        Get the input functions for the training and test set.

        :return: (list of input functions for the training set, list of input functions for the test set)
        """
        return self._get_input_fn('train'), self._get_input_fn('test')
