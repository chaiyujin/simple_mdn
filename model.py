from __future__ import absolute_import

import sys
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.distributions import Categorical
from tensorflow.contrib.distributions import Mixture
from tensorflow.contrib.distributions import MultivariateNormalDiag
from utils import process_bar


def reshape_gmm_tensor(tensor, D, K):
    tmp = []
    for i in range(D):
        tmp.append(tensor[:, :, i * K: (i + 1) * K])
    stacked = tf.stack(tmp, axis=3)
    reshaped = tf.reshape(
        stacked,
        [-1, K, D])
    return reshaped


def parameter_layer(X, Dims, K):
    locs = fully_connected(X, K * Dims, activation_fn=None)
    scales = fully_connected(X, K * Dims, activation_fn=tf.exp)
    pi = fully_connected(X, K, activation_fn=tf.nn.softmax)
    # reshape the output tensor into parameters
    locs = reshape_gmm_tensor(locs, Dims, K)
    scales = reshape_gmm_tensor(scales, Dims, K)
    pi = tf.reshape(pi, [-1, K])
    # print(locs.shape)
    # print(scales.shape)
    # print(pi.shape)
    return locs, scales, pi


def mixture(locs, scales, pi, K):
    cat = Categorical(probs=pi)
    components = [
        MultivariateNormalDiag(loc=locs[:, i], scale_diag=scales[:, i])
        for i in range(K)]
    # get the mixture distribution
    mix = Mixture(cat=cat, components=components)
    return mix


def loss_fn(y, mixture):
    loss = tf.reduce_mean(-tf.log(mixture.prob(y)))
    return loss


class Model():
    def __init__(self, config):
        # config the parameters
        self._audio_num_features = config['audio_num_features']
        self._anime_num_features = config['anime_num_features']
        self._audio_lstm_size = config['audio_lstm_size']
        self._anime_lstm_size = config['anime_lstm_size']
        self._dense_size = config['dense_size']
        self._mdn_dims = self._anime_num_features  # same as anime feature
        self._mdn_K = config['mdn_K']
        self._train = bool(config['train'])
        self._phoneme_classes = config['phoneme_classes']
        self._dropout = float(config['dropout'])

        # config the initializer
        self._initializer = tf.truncated_normal_initializer(
            mean=0., stddev=.075, seed=None, dtype=tf.float32
        )
        self._W = {}
        self._b = {}

        # set the input tensor
        self._audio_input = tf.placeholder(
            dtype=tf.float32,
            shape=[None, None, self._audio_num_features]
        )
        self._anime_data = tf.placeholder(
            dtype=tf.float32,
            shape=[None, None, self._anime_num_features]
        )
        self._seq_len = tf.placeholder(tf.int32, [None])
        # set the last_anime_data
        self._audio_input_shape = tf.shape(self._audio_input)
        self._batch_size = self._audio_input_shape[0]
        self._time_step = self._audio_input_shape[1]
        self._zero_anime_data = tf.zeros(
            [self._batch_size, 1, self._anime_num_features], dtype=tf.float32)
        self._last_anime_data = tf.concat(
            [self._zero_anime_data, self._anime_data[:, 0: -1, :]], 1
        )

        # I. build the Audio process layers
        # 1. uni-lstm layer
        self._audio_lstm = self.LSTM_layer(
            lstm_size=self._audio_lstm_size,
            inputs=self._audio_input,
            seq_len=self._seq_len,
            scope='audio_lstm'
        )
        # 2. dense layer
        # a. flatten the batch and timestep
        self._audio_lstm_output = tf.reshape(
            self._audio_lstm['output'],
            [-1, self._audio_lstm_size]
        )
        # b. xw+b
        with tf.variable_scope('lstm_phn'):
            self._W['lstm_phn'] = tf.get_variable(
                "lstm_phn_w",
                [self._audio_lstm_size, self._phoneme_classes],
                initializer=self._initializer)
            self._b['lstm_phn'] = tf.get_variable(
                "lstm_phn_b", [self._phoneme_classes],
                initializer=self._initializer)
        self._phn_vector = tf.nn.softmax(tf.nn.xw_plus_b(
            self._audio_lstm_output,
            self._W['lstm_phn'],
            self._b['lstm_phn']
        ))
        # c. reshape the vector back to batch and timestep
        self._phn_vector = tf.reshape(
            self._phn_vector,
            [self._batch_size, -1, self._phoneme_classes]
        )

        # II. build feature process layer
        self._feature_input = tf.concat(
            [self._last_anime_data, self._phn_vector], 2)
        self._feature_size = self._anime_num_features + self._phoneme_classes
        # 1. uni-lstm
        self._anime_lstm = self.LSTM_layer(
            lstm_size=self._anime_lstm_size,
            inputs=self._feature_input,
            seq_len=self._seq_len,
            scope='anime_lstm'
        )
        # 2. dense layer for mdn param
        self._anime_lstm_output = tf.reshape(
            self._anime_lstm['output'],
            [-1, self._anime_lstm_size]
        )
        with tf.variable_scope('lstm_h'):
            self._W['lstm_h'] = tf.get_variable(
                "lstm_h_w",
                [self._anime_lstm_size, self._dense_size],
                initializer=self._initializer)
            self._b['lstm_h'] = tf.get_variable(
                "lstm_h_b", [self._dense_size],
                initializer=self._initializer)
        self._dense_output = tf.nn.softmax(tf.nn.xw_plus_b(
            self._anime_lstm_output,
            self._W['lstm_h'],
            self._b['lstm_h']
        ))
        self._dense_output = tf.reshape(
            self._dense_output,
            [self._batch_size, -1, self._dense_size]
        )

        # III. mdn layer
        self._locs, self._scales_diag, self._pi = parameter_layer(
            self._dense_output, self._mdn_dims, self._mdn_K
        )
        self._mixtures = mixture(
            self._locs, self._scales_diag, self._pi, self._mdn_K
        )
        self._y = tf.reshape(
            self._anime_data,
            [-1, self._anime_num_features]
        )

        # IV. Loss function
        self._loss_fn = loss_fn(self._y, self._mixtures)

    def LSTM_layer(self, lstm_size, inputs, seq_len, scope):
        layer = {}
        layer['cell'] = self.LSTM_cell(lstm_size)
        # the zero state
        layer['zero_state'] = layer['cell'].zero_state(
            batch_size=self._batch_size, dtype=tf.float32
        )
        # last state is set as zero at first
        layer['last_state'] = layer['cell'].zero_state(
            batch_size=self._batch_size, dtype=tf.float32
        )
        if self._train:
            layer['init_state'] = layer['zero_state']
        else:
            layer['init_state'] = layer['last_state']
        layer['output'], layer['last_state'] = tf.nn.dynamic_rnn(
            cell=layer['cell'], inputs=inputs, sequence_length=seq_len,
            scope=scope, dtype=tf.float32, initial_state=layer['init_state']
        )
        return layer

    def LSTM_cell(self, size):
        cell = tf.contrib.rnn.LSTMCell(
            size, state_is_tuple=True,
            initializer=self._initializer
        )
        if self._train and self._dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=self._dropout
            )
        return cell

    def simple_train(
            self, audio, anime, seq_len,
            epoches, mini_batch_size, optimizer):
        optimizer = optimizer.minimize(self._loss_fn)
        total_samples = len(audio)
        batches = int((total_samples - 1) / mini_batch_size) + 1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epoches):
                # show the epoch number
                prefix = '\033[01;33m[Epoch ' + str(epoch) + '/' +\
                         str(epoches) + ']\033[0m'
                # training phase
                avg_train_loss = 0
                for batch in range(batches):
                    indexes = [i % total_samples
                               for i in range(batch * mini_batch_size, 
                                              (batch + 1) * mini_batch_size)]
                    bar = process_bar.process_bar(batch, batches)
                    # train batch
                    feed_train = {
                        self._audio_input: audio[indexes],
                        self._anime_data: anime[indexes],
                        self._seq_len: seq_len[indexes]
                    }
                    train_loss, _ = sess.run(
                        [self._loss_fn, optimizer],
                        feed_dict=feed_train
                    )
                    avg_train_loss += train_loss
                    train_loss_str = "%.4f" % train_loss
                    bar = prefix + bar
                    bar += ' Train Loss: ' + train_loss_str + '\r'
                    sys.stdout.write(bar)
                    sys.stdout.flush()
                # valid
                # over
                avg_train_loss = avg_train_loss / batches
                print('\nTrain Loss: ' + str(avg_train_loss))


if __name__ == '__main__':
    config = {
        'audio_num_features': 1,
        'anime_num_features': 2,
        'audio_lstm_size': 10,
        'anime_lstm_size': 10,
        'dense_size': 10,
        'mdn_K': 3,
        'phoneme_classes': 3,
        'train': 1,
        'dropout': 0.5
    }
    model = Model(config)
    audio_input = [
        [[0], [0], [0]],
        [[1], [1], [1]],
        [[2], [2], [2]],
        [[3], [3], [3]]
    ]
    anime_input = [
        [[1, 1], [1, 1], [1, 1]],
        [[1, 1], [2, 2], [3, 3]],
        [[1, 1], [2, 2], [3, 3]],
        [[1, 1], [2, 2], [3, 3]]
    ]
    seq_len = [
        3, 3, 3, 3
    ]
    session = tf.InteractiveSession()
    feed_dict = {
        model._audio_input: audio_input,
        model._anime_data: anime_input,
        model._seq_len: seq_len
    }

    optimizer = tf.train.MomentumOptimizer(1e-4, 0.9).minimize(model._loss_fn)

    tf.global_variables_initializer().run()
    epoches = 10000
    for epoch in range(epoches):
        bar = process_bar.process_bar(epoch, epoches)
        loss, _ = session.run([model._loss_fn, optimizer], feed_dict=feed_dict)
        bar += ' Loss:' + str(loss) + '\r'
        sys.stdout.write(bar)
        sys.stdout.flush()
    sys.stdout.write('\n')
