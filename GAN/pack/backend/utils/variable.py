import tensorflow as tf


def weight_variable(shape, **kwarg):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape, **kwarg):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
