from __future__ import absolute_import

import os
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.distributions import Categorical
from tensorflow.contrib.distributions import Mixture
from tensorflow.contrib.distributions import MultivariateNormalDiag
from utils import process_bar, console, format_time
from utils.media import sample_video
import data_process


def reshape_gmm_tensor(tensor, D, K):
    tmp = []
    for i in range(D):
        tmp.append(tensor[:, :, i * K: (i + 1) * K])
    stacked = tf.stack(tmp, axis=3)
    reshaped = tf.reshape(
        stacked,
        [-1, K, D])
    return reshaped


def parameter_layer(X, Dims, K, bias=0):
    locs = fully_connected(X, K * Dims, activation_fn=None)
    scales_hat = fully_connected(X, K * Dims, activation_fn=None)
    pi_hat = fully_connected(X, K, activation_fn=None)
    # add bias on the parameter
    # larger bias, more stable
    if bias > 0:
        b_scale = tf.constant(
            np.full((K * Dims), -bias, np.float32),
            dtype=tf.float32
        )
        b_pi = tf.constant(
            np.full((K), 1 + bias, np.float32),
            dtype=tf.float32
        )
        scales_hat = tf.add(scales_hat, b_scale)
        pi_hat = tf.multiply(pi_hat, b_pi)
    scales = tf.exp(scales_hat)
    pi = tf.nn.softmax(pi_hat)
    # reshape the output tensor into parameters
    locs = reshape_gmm_tensor(locs, Dims, K)
    scales = reshape_gmm_tensor(scales, Dims, K)
    pi = tf.reshape(pi, [-1, K])
    return locs, scales, pi


def mixture(locs, scales, pi, K):
    cat = Categorical(probs=pi)
    components = [
        MultivariateNormalDiag(loc=locs[:, i, :], scale_diag=scales[:, i, :])
        for i in range(K)]
    # get the mixture distribution
    mix = Mixture(cat=cat, components=components)
    return mix


def sample_gmm(locs, scales, pi):
    idx = np.random.choice(pi.shape[1], p=pi[0])
    loc = locs[0, idx, :]
    scale = scales[0, idx, :]
    scale_diag = np.zeros((scale.shape[0], scale.shape[0]))
    for i in range(len(scale)):
        scale_diag[i][i] = scale[i]
    x = np.random.multivariate_normal(loc, scale_diag, 1)
    return x[0]


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
        self._mdn_bias = float(config['mdn_bias'])
        if self._train:
            self._mdn_bias = 0

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
        if self._train:
            self._last_anime_data = tf.concat(
                [self._zero_anime_data, self._anime_data[:, 0: -1, :]], 1
            )
        else:
            # when no train, directly use anime_data
            self._last_anime_data = self._anime_data

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
              self._dense_output, self._mdn_dims, self._mdn_K, self._mdn_bias
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

        # saver
        self._saver = tf.train.Saver()

    def LSTM_layer(self, lstm_size, inputs, seq_len, scope):
        layer = {}
        layer['cell'] = self.LSTM_cell(lstm_size)
        layer['init_state'] = layer['cell'].zero_state(
            batch_size=self._batch_size, dtype=tf.float32
        )
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

    def run_one_epoch(
            self, sess, data, batch_size,
            optimizer, log_prefix=''):
        total_samples = len(data['inputs'])
        batches = int((total_samples - 1) / batch_size) + 1
        avg_loss = 0
        for batch in range(batches):
            indexes = [i % total_samples
                       for i in range(batch * batch_size,
                                      (batch + 1) * batch_size)]
            bar = process_bar.process_bar(batch, batches)
            # feed dict
            feed = {
                self._audio_input: data['inputs'][indexes],
                self._anime_data: data['outputs'][indexes],
                self._seq_len: data['seq_len'][indexes]
            }
            if optimizer is not None:
                loss, _ = sess.run(
                    [self._loss_fn, optimizer],
                    feed_dict=feed
                )
            else:
                loss = sess.run(self._loss_fn, feed_dict=feed)
            avg_loss += loss
            loss_str = "%.4f" % loss
            if optimizer is not None:
                bar += ' Train Loss: ' + loss_str + '\r'
            else:
                bar += ' Valid Loss: ' + loss_str + '\r'
            console.log('log', log_prefix, bar)
        print()
        avg_loss /= batches
        return avg_loss

    def simple_train(
            self, epoches, optimizer,
            train_data, mini_batch_size,
            valid_data=None, valid_batch_size=None):
        # fix the batch size
        if valid_batch_size is None:
            valid_batch_size = mini_batch_size
        if mini_batch_size > len(train_data['inputs']):
            mini_batch_size = len(train_data['inputs'])
        if valid_batch_size > len(valid_data['inputs']):
            valid_batch_size = len(valid_data['inputs'])
        # begin to train
        epoch_list = []
        train_loss_list = []
        valid_loss_list = []
        optimizer = optimizer.minimize(self._loss_fn)
        # best loss
        best_valid_loss = 1000000
        # time
        total_time = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epoches):
                # print epoch
                console.log(
                    'log',
                    'Epoch ' + str(epoch) + '/' + str(epoches), '\n')
                # start time
                start_time = time.time()
                # show the epoch number
                prefix = 'Train'
                # training phase
                train_loss = self.run_one_epoch(
                    sess, train_data,
                    mini_batch_size, optimizer, prefix)
                # validation
                valid_prefix = 'Valid'
                valid_loss = self.run_one_epoch(
                    sess, valid_data, valid_batch_size, None, valid_prefix
                )
                # put into list
                epoch_list.append(epoch)
                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)
                # end time
                delta_time = time.time() - start_time
                total_time += delta_time
                avg_time = total_time / (epoch + 1)
                need_time = avg_time * (epoches - epoch - 1)
                delta_time = format_time.format_sec(delta_time)
                need_time = format_time.format_sec(need_time)
                console.log('log', 'Time', delta_time + '\n')
                console.log('log', 'Left', need_time + '\n')
                # console the training loss
                console.log('info', 'Train Loss', str(train_loss) + '\n')
                console.log('info', 'Valid Loss', str(valid_loss) + '\n')
                print()

                # save the model
                if valid_loss < best_valid_loss:
                    if (epoch + 1) % 10 == 0 or\
                       (epoch + 1) == epoches or\
                       epoch > 400:
                        best_valid_loss = valid_loss
                        self.save(sess, epoch)
                # draw the figure of the training process
                if (epoch + 1) % 10 == 0 or (epoch + 1) == epoches:
                    fig = plt.figure(figsize=(12, 12))
                    cost_plt = fig.add_subplot(111)
                    cost_plt.title.set_text('Cost')
                    cost_plt.plot(
                        epoch_list, train_loss_list, 'g',
                        epoch_list, valid_loss_list, 'r')
                    plt.savefig('error.png')
                    plt.clf()

    def sample_one_step(self, sess, audio_frame, anime_frame):
        assert(audio_frame.shape[0] == 1)
        assert(audio_frame.shape[1] == 1)
        assert(audio_frame.shape[2] == self._audio_num_features)
        feed_dict = {
            self._audio_input: audio_frame,
            self._anime_data: anime_frame,
            self._seq_len: [1]
        }
        if self._audio_lstm['next_state'] is not None and\
           self._anime_lstm['next_state'] is not None:
            # set next state of audio lstm
            feed_dict[self._audio_lstm['init_state']] =\
                self._audio_lstm['next_state']
            # set next state of anime lstm
            feed_dict[self._anime_lstm['init_state']] =\
                self._anime_lstm['next_state']
        # feed
        is0, is1,\
            self._audio_lstm['next_state'],\
            self._anime_lstm['next_state'],\
            locs, scales, pi = sess.run(
                [
                    self._audio_lstm['init_state'],
                    self._anime_lstm['init_state'],
                    self._audio_lstm['last_state'],
                    self._anime_lstm['last_state'],
                    self._locs, self._scales_diag, self._pi
                ],
                feed_dict=feed_dict
            )
        # print(is0)
        # print(is1)
        # os.system('pause')
        return sample_gmm(locs, scales, pi)

    def sample_audio(self, sess, audio_input):
        assert(audio_input.shape[0] == 1)
        assert(audio_input.shape[1] >= 1)
        assert(audio_input.shape[2] == self._audio_num_features)
        anime_data = []
        anime_frame = np.zeros((1, 1, self._anime_num_features))
        # init next_state
        self._anime_lstm['next_state'] = None
        self._audio_lstm['next_state'] = None
        for time_step in range(audio_input.shape[1]):
            audio_frame = audio_input[:, time_step: time_step + 1, :]
            anime_frame = self.sample_one_step(
                sess, audio_frame, anime_frame)
            anime_data.append(anime_frame)
            anime_frame = anime_frame.reshape(1, 1, self._anime_num_features)
        return anime_data

    def sample_data(self, sess, data, number=None, video=False):
        if number is None or number > len(data['inputs']):
            number = len(data['inputs'])
        avg_mse = 0
        avg_value = 0
        count = number
        print(number)
        for idx in range(number):
            bar = process_bar.process_bar(idx, number)
            console.log('log', 'Sample', bar + '\r')
            audio = data['inputs'][idx: idx + 1]
            anime_true = data['outputs'][idx]
            assert(len(audio[0]) == len(anime_true))
            path_prefix = data['path_prefix'][idx]

            anime_pred = self.sample_audio(sess, audio)
            assert(len(anime_pred) == len(anime_true))
            # update error
            zeros = np.zeros(anime_true.shape[0])
            mse = ((anime_pred - anime_true) ** 2).mean()
            avg_value += ((anime_true - zeros) ** 2).mean()
            avg_mse += mse

            if video:
                anime_pred, _ = data_process.pad_sequences(
                    anime_pred, 19
                )
                anime_true, _ = data_process.pad_sequences(
                    anime_true, 19
                )
                sample_video(
                    {
                        'path_prefix': path_prefix,
                        'anime_pred': anime_pred,
                        'anime_true': anime_true
                    },
                    'result/' + str(idx) + '.mp4'
                )
        console.log()
        # return the mean square error and avg value
        if count == 0:
            return None
        else:
            avg_value /= count
            avg_mse /= count
        return avg_mse / avg_value

    def load(self, sess, path='./model/best'):
        path = os.path.abspath(path)
        self._saver.restore(sess, path)

    def save(self, sess, epoch=0, path='./model/best'):
        path = os.path.abspath(path)
        self._saver.save(sess, path)


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

