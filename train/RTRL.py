from __future__ import absolute_import

import sys
sys.path.append('..')
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import console, process_bar, format_time

class Trainer():
    def __init__(self, model, train_set, valid_set):
        self._model = model
        self._train_set = train_set
        self._valid_set = valid_set

    def train(self, optimizer, epoches, mini_batch_size, valid_batch_size):
        with tf.Session() as self._sess:
            self._optimizer = optimizer.minimize(self._model._loss_fn)
            self._sess.run(tf.global_variables_initializer())
            epoch_list = []
            train_loss_list = []
            valid_loss_list = []
            error_rate_list = []
            total_time = 0
            for epoch in range(epoches):
                start_time = time.time()
                console.log('log', 'Epoch ' + str(epoch) + '/' + str(epoches))
                # training by time
                train_loss = self.run_one_epoch(
                    data_set=self._train_set,
                    batch_size=mini_batch_size,
                    is_train=True
                )
                # valid by time
                valid_loss = self.run_one_epoch(
                    data_set=self._valid_set,
                    batch_size=valid_batch_size,
                    is_train=False
                )
                error_rate = self._model.sample_data(
                    self._sess, self._valid_set, valid_batch_size)

                epoch_list.append(epoch)
                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)
                if error_rate > 1:
                    error_rate_list.append(1)
                else:
                    error_rate_list.append(error_rate)
                console.log('info', 'Train Loss', str(train_loss) + '\n')
                console.log('info', 'Valid Loss', str(valid_loss) + '\n')
                console.log('info', 'Error Rate', str(error_rate) + '\n')

                if True:
                    fig = plt.figure(figsize=(12, 12))
                    cost_plt = fig.add_subplot(211)
                    rate_plt = fig.add_subplot(212)
                    cost_plt.title.set_text('Cost')
                    cost_plt.plot(
                        epoch_list, train_loss_list, 'g',
                        epoch_list, valid_loss_list, 'r'
                    )
                    rate_plt.plot(
                        epoch_list, error_rate_list, 'r'
                    )
                    plt.savefig('error.png')
                    plt.clf()

                # end time
                delta_time = time.time() - start_time
                total_time += delta_time
                avg_time = total_time / (epoch + 1)
                need_time = avg_time * (epoches - epoch - 1)
                delta_time = format_time.format_sec(delta_time)
                need_time = format_time.format_sec(need_time)
                console.log('log', 'Epoch Time', delta_time + '\n')
                console.log('log', 'Total Left', need_time + '\n')
                # end of epoch
                console.log('log', '---End of Epoch---', '\n\n')

    def run_one_epoch(self, data_set, batch_size, is_train=True):
        if batch_size > len(data_set['inputs']):
            batch_size = len(data_set['inputs'])
        if batch_size <= 0:
            batch_size = 1
        total_samples = len(data_set['inputs'])
        batches = int((total_samples - 1) / batch_size) + 1

        avg_loss = 0
        avg_count = 0
        for batch in range(batches):
            indexes = [i % total_samples
                       for i in range(batch * batch_size,
                                      (batch + 1) * batch_size)]
            inputs = np.asarray(data_set['inputs'])[indexes]
            outputs = np.asarray(data_set['outputs'])[indexes]
            seq_len = np.asarray(data_set['seq_len'])[indexes]
            length = np.amax(seq_len)

            result = None
            for step in range(length):
                input = inputs[:, step: step + 1, :]
                output = outputs[:, step: step + 1, :]
                seq_l = np.full(seq_len.shape, 1, dtype=np.int32)
                feed_dict = {
                    self._model._inputs: input,
                    self._model._outputs: output,
                    self._model._seq_len: seq_l
                }
                # generate the to_run list
                to_run = [self._model._loss_fn]
                # append lstm layers
                for i, lstm in enumerate(self._model._lstm_layers):
                    to_run.append(lstm['last_state'])
                    if result is not None:
                        feed_dict[lstm['init_state']] = result[i + 1]
                if is_train:
                    to_run.append(self._optimizer)
                    result = self._sess.run(
                        to_run,
                        feed_dict=feed_dict
                    )
                else:
                    result = self._sess.run(
                        to_run,
                        feed_dict=feed_dict
                    )
                avg_loss += result[0]
                avg_count += 1

            bar = process_bar.process_bar(batch, batches)
            loss_str = '\tLoss %.4f' % (avg_loss / avg_count)
            if is_train:
                console.log('', 'Train', bar + loss_str + '\r')
            else:
                console.log('', 'Valid', bar + loss_str + '\r')
        console.log()
        return avg_loss / avg_count


