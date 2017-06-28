import tensorflow as tf
import tensorflow.contrib.layers as layers


def lrelu(x, slope=0.2):
    with tf.name_scope('lrelu'):
        x = tf.identity(x)
        return (0.5 * (1 + slope)) * x + (0.5 * (1 - slope)) * tf.abs(x)


def deconv2d(
        x, num_outputs, kernel=[4, 4], strides=[2, 2],
        activation_fn=lrelu, batch_norm=False, padding='SAME',
        weights_initializer=tf.random_normal_initializer(0, 0.02)):

    norm_fn = layers.batch_norm if batch_norm else None
    r = layers.conv2d_transpose(
            x, num_outputs, kernel, stride=strides,
            activation_fn=activation_fn,
            normalizer_fn=norm_fn,
            padding=padding,
            weights_initializer=weights_initializer
        )
    return r


def conv2d(
        x, num_outputs, kernel=[4, 4], strides=[2, 2],
        activation_fn=lrelu, batch_norm=False, padding='SAME',
        weights_initializer=tf.random_normal_initializer(0, 0.02)):

    norm_fn = layers.batch_norm if batch_norm else None
    r = layers.conv2d(
            x, num_outputs, kernel, strides,
            activation_fn=activation_fn,
            normalizer_fn=norm_fn,
            weights_initializer=weights_initializer
        )
    return r

