from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from backend.nn import layer
from backend.model import BasicModel
from backend.utils.media import sample_video
from backend.utils import process_bar, console


class Model(BasicModel):
    def __init__(self, config=None):
        super(Model, self).__init__(config)
        # config input
        self._audio_num_features = config['audio_num_features']
        self._anime_num_features = config['anime_num_features']
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
            [None, None, self._anime_num_features]
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

        # define the network
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
            output_size=self._anime_num_features,
            inputs=self._phn_layer['output'],
            initializer=self._initializer,
            activation=None,
            scope='output_dense'
        )

        self._anime_pred = tf.nn.sigmoid(self._output_layer['output'])

        self._loss_fn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self._output_layer['output'],
            labels=tf.reshape(self._outputs, [-1, self._anime_num_features])
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
        return self._anime_pred

    def error_rate(self, pred, true):
        total = 1
        for x in pred.shape:
            total *= x
        error = ((pred.flatten() - true.flatten()) ** 2).sum()
        return error / total

    def sample_one_step(self, sess, audio_frame, anime_frame):
        bs = audio_frame.shape[0]
        assert(audio_frame.shape[1] == 1)
        assert(audio_frame.shape[2] == self._audio_num_features)
        feed_dict = {
            self._inputs: audio_frame,
            self._outputs: anime_frame,
            self._seq_len: np.full((bs), 1, np.int32)
        }
        to_run = [self.pred_tensor]
        for lstm in self._lstm_layers:
            if lstm['next_state'] is not None:
                feed_dict[lstm['init_state']] = lstm['next_state']
            to_run.append(lstm['last_state'])
        # feed
        result = sess.run(
                to_run,
                feed_dict=feed_dict
            )
        output_anime = result[0]
        for i, lstm in enumerate(self._lstm_layers):
            lstm['next_state'] = result[i + 1]
        anime_frame = []
        for i in range(bs):
            anime_frame.append([output_anime[i]])
        anime_frame = np.asarray(anime_frame, dtype=np.float32)
        return anime_frame

    def sample_audio(self, sess, audio_input):
        bs = audio_input.shape[0]
        assert(audio_input.shape[1] >= 1)
        assert(audio_input.shape[2] == self._audio_num_features)
        anime_data = None
        anime_frame = np.zeros((bs, 1, self._anime_num_features))
        # init next_state
        for lstm in self._lstm_layers:
            lstm['next_state'] = None
        for time_step in range(audio_input.shape[1]):
            audio_frame = audio_input[:, time_step: time_step + 1, :]
            anime_frame = self.sample_one_step(
                sess, audio_frame, anime_frame)
            if anime_data is None:
                anime_data = anime_frame
            else:
                # append on time axis
                anime_data = np.append(anime_data, anime_frame, axis=1)
            anime_frame = anime_frame.reshape(bs, 1, self._anime_num_features)
        return np.asarray(anime_data)

    def sample(self, sess, data, batch_size, number=None, video=False):
        if number is None or number > data.length:
            number = data.length
        if batch_size > number:
            batch_size = number
        batches = int(number / batch_size)
        data.batch_size = batch_size
        data.reset()
        bar = process_bar.process_bar(0, batches)
        console.log('log', 'Sample', bar + '\r')
        for idx in range(batches):
            li = idx * batch_size
            d, s = data.next_batch()
            audio = d['inputs']
            anime_true = d['outputs']
            path_prefix = d['path_prefix']

            anime_pred = self.sample_audio(sess, audio)
            assert(len(anime_pred) == len(anime_true))

            # anime_pred = anime_pred ** 2
            anime_true = anime_true ** 4

            if video:
                for i in range(batch_size):
                    pred = anime_pred[i]
                    true = anime_true[i]
                    sample_video(
                        {
                            'path_prefix': path_prefix[i],
                            'anime_pred': pred,
                            'anime_true': true
                        },
                        'result/' + str(li + i) + '.mp4'
                    )
            bar = process_bar.process_bar(idx, batches)
            console.log('log', 'Sample', bar + '\r')
        console.log()

        return
