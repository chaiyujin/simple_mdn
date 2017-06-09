from __future__ import absolute_import

import tensorflow as tf
from .ops import conv2d, deconv2d, lrelu


class Net():
    def __init__(self, scope='Net', config=None):
        self._scope = scope

    def __call__(self, x, reuse=False):
        raise NotImplementedError()

    @property
    def theta(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self._scope
        )

    def __get_activation_fn(self, activation):
        if activation == 'None':
            return None
        elif activation == 'relu':
            return tf.nn.relu
        elif activation == 'tanh':
            return tf.tanh
        elif activation == 'sigmoid':
            return tf.sigmoid
        elif activation == 'softmax':
            return tf.nn.softmax
        elif activation == 'lrelu':
            return lrelu
        else:
            raise NotImplementedError()

    def _conv_layer(
            self, list, num_outputs, kernel=[4, 4], strides=[2, 2],
            activation='None', batch_norm=False, padding='SAME',
            w_init=None, dropout=0):
        params = [
            num_outputs, kernel, strides,
            self.__get_activation_fn(activation),
            batch_norm, padding,
        ]
        if w_init is not None:
            params.append(w_init)
        list.append({
            'params': params,
            'dropout': dropout
        })

    def _deconv_layer(
            self, list, num_outputs, kernel=[4, 4], strides=[2, 2],
            activation='None', batch_norm=False, padding='SAME',
            w_init=None, dropout=0):
        params = [
            num_outputs, kernel, strides,
            self.__get_activation_fn(activation),
            batch_norm, padding,
        ]
        if w_init is not None:
            params.append(w_init)
        list.append({
            'params': params,
            'dropout': dropout
        })


class UNet(Net):
    def __init__(self, scope='UNet', config=None):
        super(UNet, self).__init__(scope, config)
        self._enc_layers = []
        self._dec_layers = []
        self._encoding = True
        self._decoding = False

    def __call__(self, x, reuse=False):
        # assert(self._decoding)
        layers = []
        with tf.variable_scope(self._scope, reuse=reuse):
            num_encodes = len(self._enc_layers)
            num_decodes = len(self._dec_layers)
            for i, layer in enumerate(self._enc_layers):
                last = x if i == 0 else layers[-1]
                v_scope = 'encode_' + str(i)
                if i == num_encodes - 1:
                    v_scope = 'bottleneck'
                with tf.variable_scope(v_scope):
                    res = conv2d(last, *layer['params'])
                    layers.append(res)

            for i, layer in enumerate(self._dec_layers):
                mirror = layers[num_encodes - 1 - i]
                last = layers[-1]
                if i > 0:
                    last = tf.concat([last, mirror], 3)
                v_scope = 'decode_' + str(num_encodes - 2 - i)
                if i == num_decodes - 1:
                    v_scope = 'output'
                with tf.variable_scope(v_scope):
                    res = deconv2d(last, *layer['params'])
                    if layer['dropout'] > 0:
                        res = tf.nn.dropout(
                            res, keep_prob=1 - layer['dropout']
                        )
                    layers.append(res)

        return layers[-1]

    def encode_layer(
            self, num_outputs, kernel=[4, 4], strides=[2, 2],
            activation='None', batch_norm=False, padding='SAME',
            w_init=None, dropout=0):
        assert(self._encoding)
        self._conv_layer(
            self._enc_layers, num_outputs, kernel, strides,
            activation, batch_norm, padding, w_init, dropout)

    def decode_layer(
            self, num_outputs, kernel=[4, 4], strides=[2, 2],
            activation='None', batch_norm=False, padding='SAME',
            w_init=None, dropout=0):
        self._encoding = False
        self._decoding = True
        self._deconv_layer(
            self._dec_layers, num_outputs, kernel, strides,
            activation, batch_norm, padding, w_init, dropout)


class ConvNet(Net):
    def __init__(self, scope='ConvNet', config=None):
        super(ConvNet, self).__init__(scope, config)
        self._layers = []

    def __call__(self, x, reuse=False):
        layers = []
        with tf.variable_scope(self._scope, reuse=reuse):
            for i, layer in enumerate(self._layers):
                last = x if i == 0 else layers[-1]
                v_scope = 'conv_' + str(i)
                if i + 1 == len(self._layers):
                    v_scope = 'output'
                with tf.variable_scope(v_scope):
                    res = conv2d(last, *layer['params'])
                    if layer['dropout'] > 0:
                        res = tf.nn.dropout(
                            res, keep_prob=1 - layer['dropout']
                        )
                    layers.append(res)

        return layers[-1]

    def conv_layer(
            self, num_outputs, kernel=[4, 4], strides=[2, 2],
            activation='None', batch_norm=False, padding='SAME',
            w_init=None, dropout=0):
        self._conv_layer(
            self._layers, num_outputs, kernel, strides,
            activation, batch_norm, padding, w_init, dropout)


class DeconvNet(Net):
    def __init__(self, scope='DeconvNet', config=None):
        super(DeconvNet, self).__init__(scope, config)
        self._layers = []

    def __call__(self, x, reuse=False):
        layers = []
        with tf.variable_scope(self._scope, reuse=reuse):
            for i, layer in enumerate(self._layers):
                last = x if i == 0 else layers[-1]
                v_scope = 'deconv_' + str(i)
                if i + 1 == len(self._layers):
                    v_scope = 'output'
                with tf.variable_scope(v_scope):
                    res = deconv2d(last, *layer['params'])
                    if layer['dropout'] > 0:
                        res = tf.nn.dropout(
                            res, keep_prob=1 - layer['dropout']
                        )
                    layers.append(res)

        return layers[-1]

    def deconv_layer(
            self, num_outputs, kernel=[4, 4], strides=[2, 2],
            activation='None', batch_norm=False, padding='SAME',
            w_init=None, dropout=0):
        self._deconv_layer(
            self._layers, num_outputs, kernel, strides,
            activation, batch_norm, padding, w_init, dropout)


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    import os
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

    # net = UNet()
    # net.encode_layer(1, activation='lrelu')
    # net.encode_layer(2, activation='lrelu', batch_norm=True)
    # net.decode_layer(1, activation='relu', batch_norm=True)
    # net.decode_layer(1)
    conv_net = ConvNet('ConvNet')
    conv_net.conv_layer(1, activation='lrelu')
    conv_net.conv_layer(2, activation='lrelu', batch_norm=True)
    decv_net = DeconvNet('Deconv')
    decv_net.deconv_layer(1, activation='relu', batch_norm=True)
    decv_net.deconv_layer(1)

    bs = 8
    x = tf.placeholder(
        tf.float32,
        [bs, 28, 28, 1]
    )

    bn = conv_net(x)
    y_logits = decv_net(bn)
    # y_logits = net(x)
    y = tf.sigmoid(y_logits)

    loss_fn = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=x,
            logits=y_logits
        )
    )
    optim = tf.train.AdamOptimizer().minimize(loss_fn)

    tf.summary.scalar('cross_entropy', loss_fn)

    def plot(samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        return fig

    i = 0
    if not os.path.exists('out'):
        os.mkdir('out')
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./log',
                                             tf.get_default_graph())
        sess.run(tf.global_variables_initializer())
        for epoch in range(100000):
            x_b, _ = mnist.train.next_batch(bs)
            x_b = np.reshape(
                x_b,
                [bs, 28, 28, 1]
            )
            # sample
            if epoch % 1000 == 0:
                samples = sess.run(y, feed_dict={x: x_b})
                samples = np.append(samples, x_b, axis=0)
                fig = plot(samples)
                plt.savefig(
                    'out/{}.png'.format(str(i).zfill(3)),
                    bbox_inches='tight')
                i += 1
                plt.close(fig)
            summary, loss, _ = sess.run(
                [merged, loss_fn, optim],
                feed_dict={x: x_b}
            )
            train_writer.add_summary(summary, epoch)

            info = 'Epoch ' + str(epoch) + ': '
            info += 'loss %.4f \r' % loss
            sys.stdout.write(info)
            sys.stdout.flush()
