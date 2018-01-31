import numpy as np
from tensorflow.python.keras.backend import set_learning_phase

from utils.misc import *
from utils.models import OrderDependentModel


def model_fn(features, labels, mode, params):
    np.random.seed(params['np_seed'])
    model = OrderDependentModel(params['layer_config'])

    if mode == tf.estimator.ModeKeys.PREDICT:
        set_learning_phase(False)
        score = model(features)
        predictions = {'score': score}
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'score': tf.estimator.export.PredictOutput(predictions)
            })

    if mode == tf.estimator.ModeKeys.TRAIN:
        set_learning_phase(True)
        labels = labels['label']

        logits = model(features)
        loss = model.compute_loss(labels, logits)
        train_op = model.get_train_op(loss)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        set_learning_phase(False)
        labels = labels['label']

        logits = model(features)
        loss = model.compute_loss(labels, logits)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss)


tf.logging.set_verbosity(tf.logging.INFO)

# layer_config can be one of [256], [256, 512], [256, 256, 512] or [256, 256, 512, 512] to replicate the 1, 2, 3 or 4
# layer version respectively.
# the experiment's result (graph, checkpoint, csv of metrics) will be saved in the model_dir/model_name directory
params = {'layer_config': [256], 'model_name': 'order_dependent_1_layer',
          'model_dir': '../model_dir/order_dependent'}
experiment = Experiment(n_epochs=80, n_experiments=10)
results = experiment.run_experiment(model_fn, params)
print('Experiment result is:\n{}'.format(results))
