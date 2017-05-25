from __future__ import division

import numpy as np
import tensorflow as tf
from backend.nn import layer
from backend.model import BasicModel


class Model(BasicModel):
    def __init__(self, config=None):
        super(Model, self).__init__(config)
        # config input
        self._audio_num_features = config['audio_num_features']
        self._silence_label_size = 1
        self._audio_lstm_size = config['audio_lstm_size']
        self._audio_lstm_dropout = config['audio_lstm_dropout']
        self._phoneme_classes = config['phoneme_classes']
        self._dense_size = config['dense_size']
        self._train = bool(config['train'])
        self._dropout = float(config['dropout'])
        if not self._train:
            self._dropout = 0
            self._audio_lstm_dropout = 0

        self._inputs = tf.placeholder(
            tf.float32,
            [None, None, self._audio_num_features]
        )
        self._outputs = tf.placeholder(
            tf.float32,
            [None, None]
        )
        self._seq_len = tf.placeholder(
            tf.int32,
            [None]
        )
        self._placeholder_dict = {
            'inputs': self._inputs,
            'outputs': self._outputs,
            'seq_len': self._seq_len
        }
        self._initializer = tf.truncated_normal_initializer(
            mean=0., stddev=.075, seed=None, dtype=tf.float32
        )
        self._audio_input_shape = tf.shape(self._inputs)
        self._batch_size = self._audio_input_shape[0]
        self._time_step = self._audio_input_shape[1]
        self._lstm_layers = []
        # then define the network
        self._audio_lstm = layer.LSTM_layer(
            batch_size=self._batch_size,
            lstm_size=self._audio_lstm_size,
            inputs=self._inputs,
            seq_len=self._seq_len,
            dropout=self._audio_lstm_dropout,
            scope='audio_lstm'
        )
        self._lstm_layers.append(self._audio_lstm)
        self._audio_lstm_output = tf.reshape(
            self._audio_lstm['output'],
            [-1, self._audio_lstm_size]
        )
        self._phn_layer = layer.dense_layer(
            input_size=self._audio_lstm_size,
            output_size=self._phoneme_classes,
            inputs=self._audio_lstm_output,
            initializer=self._initializer,
            activation=tf.nn.relu,
            scope='lstm2phn_dense'
        )
        self._output_layer = layer.dense_layer(
            input_size=self._phoneme_classes,
            output_size=self._silence_label_size,
            inputs=self._phn_layer['output'],
            initializer=self._initializer,
            activation=None,
            scope='output_dense'
        )

        self._silence_pred = tf.nn.sigmoid(self._output_layer['output'])

        self._loss_fn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self._output_layer['output'],
            labels=tf.reshape(self._outputs, [-1, self._silence_label_size])
        ))
        # finally, create the saver
        self._saver = tf.train.Saver()

    @property
    def placeholder_dict(self):
        return self._placeholder_dict

    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def pred_tensor(self):
        return self._silence_pred

    def error_rate(self, pred, true):
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        total = 1
        for x in pred.shape:
            total *= x
        error = np.abs(pred.flatten() - true.flatten()).sum()
        return error / total

