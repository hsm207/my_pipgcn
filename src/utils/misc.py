import datetime
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from datasets.NoDiffusionDataset import NoDiffusionDataset
from utils.metrics import AUC_Calculator


def current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')


class Experiment:
    """
    A class to train and evaluate a model as per the parameters in the paper.

    """

    def __init__(self, n_experiments=10, n_epochs=80, n_samples=-1, n_input_fns=None):
        """

        :param n_experiments: An integer between 1 and 10 representing the number of times to run the model (train
               and evaluate)
        :param n_epochs: Number of epochs for training
        :param n_samples: Number of paired examples to use for each protein in both the training and test set. Set to
               -1 to use all paired examples
        :param n_input_fns: Number of input functions to use for the training set and test set (each protein has its
                            own input function). Set to None to use all the input functions.
        """

        # seeds are from original author's code
        seeds = [
            {"tf_seed": 649737, "np_seed": 29820},
            {"tf_seed": 395408, "np_seed": 185228},
            {"tf_seed": 252356, "np_seed": 703889},
            {"tf_seed": 343053, "np_seed": 999360},
            {"tf_seed": 743746, "np_seed": 67440},
            {"tf_seed": 175343, "np_seed": 378945},
            {"tf_seed": 856516, "np_seed": 597688},
            {"tf_seed": 474313, "np_seed": 349903},
            {"tf_seed": 838382, "np_seed": 897904},
            {"tf_seed": 202003, "np_seed": 656146},
        ]

        self.seeds = seeds[:n_experiments]
        self.num_epochs = n_epochs
        self.mean_roc_auc = []

        self.train_input_fns, self.test_input_fns = NoDiffusionDataset(take=n_samples).get_input_fns()

        if n_input_fns is not None:
            self.train_input_fns = self.train_input_fns[:n_input_fns]
            self.test_input_fns = self.test_input_fns[:n_input_fns]

    def _create_config(self, params):
        model_dir = os.path.join(params['model_dir'], params['model_name'])
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        config = tf.estimator.RunConfig(model_dir=model_dir,
                                        tf_random_seed=params['tf_seed'],
                                        keep_checkpoint_max=2,
                                        save_summary_steps=1000000,
                                        save_checkpoints_steps=100000,
                                        session_config=sess_config)

        return config

    def _train_model(self, model):
        for i in range(self.num_epochs):
            for j, train_input_fn in enumerate(self.train_input_fns):
                print('{}: Epoch {}: Training on protein {}...\n'.format(current_time(), i + 1, j + 1))
                model.train(train_input_fn)

    def _test_model(self, model):
        roc_auc_calculator = AUC_Calculator()
        mean_roc_auc = roc_auc_calculator.compute_roc_auc(model, self.test_input_fns)
        self.mean_roc_auc.append(mean_roc_auc)

    def report_results(self, params):
        """
        Returns a pandas data frame containing the model name and the mean and standard deviation of the roc auc
        accross different trials.

        :param params: A ditionary of parameters containing the estimator's model_dir and model name.
        :return: A 3 column pandas data frame
        """
        savepath = os.path.join(params['model_dir'], params['model_name'], 'experiment_summary.csv')
        data = {'model': params['model_name'], 'mean_roc_auc': np.mean(self.mean_roc_auc),
                'sd_roc_auc': np.std(self.mean_roc_auc)}
        df = pd.Series(data).to_frame().transpose()[['model', 'mean_roc_auc', 'sd_roc_auc']]
        df.to_csv(savepath, header=True, index=False)
        return df

    def run_experiment(self, model_fn, params):
        """
        Execute a model_fn given its config and params.

        :param model_fn: A model_fn to be fed to an Estimator
        :param params: A dictionary containing parameters to pass to an Estimator
        :return: A pandas data frame summarizing the experiment's results
        """
        for i, seed in enumerate(self.seeds, 1):
            print('{}: Running experiment {} for model {}'.format(current_time(),
                                                                  i,
                                                                  params['model_name']))
            params.update(seed)
            config = self._create_config(params)
            pip_model = tf.estimator.Estimator(model_fn=model_fn, config=config, params=params)
            self._train_model(pip_model)
            self._test_model(pip_model)

        results_df = self.report_results(params)
        return results_df
