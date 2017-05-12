from __future__ import absolute_import

import tensorflow as tf
from nn import mdn
from utils import variable

class Model():
    def __init__(self, config):
        # config the parameters
        self.audio_num_features = config['audio_num_features']
        self.anime_num_features = config['anime_num_features']
        self.audio_lstm_size = config['audio_lstm_size']
        self.train = bool(config['train'])
        self.dropout = float(config['dropout'])

        # config the initializer
        self.initializer = tf.truncated_normal_initializer(
            mean=0., stddev=.075, seed=None, dtype=tf.float32
        )

        # set the input tensor
        self.audio_input = tf.placeholder(
            dtype=tf.float32,
            shape=[None, None, self.audio_num_features]
        )
        self.anime_data = tf.placeholder(
            dtype=tf.float32,
            shape=[None, None, self.anime_num_features]
        )
        self.seq_len = tf.placeholder(tf.int32, [None])

        # build the Audio process layers
        self.audio_lstm_cell = self.LSTM_cell(self.audio_lstm_size)
        self.phn_vector, _ = tf.nn.dynamic_rnn(
            self.audio_lstm_cell, self.audio_input,
            self.seq_len, dtype=tf.float32
        )
        self.feature_input = tf.concat([self.anime_data, self.phn_vector], 2)
        # self.output = 

    def LSTM_cell(self, size):
        cell = tf.contrib.rnn.LSTMCell(
            size, state_is_tuple=True,
            initializer=self.initializer
        )
        if self.train and self.dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=self.dropout
            )
        return cell
