import tensorflow as tf
import numpy as np

def initializer(init, shape):
    if init == "zero":

        return tf.zeros(shape)

    elif init == "he":
        # TODO: Figure out formula's derivation
        # The paper: https://arxiv.org/pdf/1502.01852.pdf
        fan_in = np.prod(shape)

        std = 1 / np.sqrt(fan_in)

        return tf.random_uniform(shape, minval=-std, maxval=std, dtype=tf.float64)
