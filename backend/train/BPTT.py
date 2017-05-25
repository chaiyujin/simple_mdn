from __future__ import absolute_import
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .base import BasicTrainer
from ..utils import console, process_bar, format_time


class Trainer(BasicTrainer):
    def __init__(self, model, train_set, valid_set, label_key, feed_keys=None):
        super(Trainer, self).__init__(
            model, train_set, valid_set, label_key, feed_keys)

    def checkpoint_epoch(self, epoch, epoches, cp=10):
        if (epoch + 1) % cp == 0 or (epoch + 1) == epoches:
            return True
        else:
            return False

    def train(self, sess, optimizer, epoches,
              mini_batch_size, valid_batch_size,
              early_stopping_n=-1, load=False):
        if sess is not None:
            self._sess = sess
            self._optimizer = optimizer.minimize(self._model._loss_fn)
            self._sess.run(tf.global_variables_initializer())
            if load:
                self._model.load(self._sess)
            self._path = self._model._default_path
            epoch_list = []
            train_loss_list = []
            valid_loss_list = []
            train_er_list = []
            valid_er_list = []
            total_time = 0
            best_valid_loss = 1000000
            early_stopping_cnt = 0
            for epoch in range(epoches):
                start_time = time.time()
                console.log('log', 'Epoch ' + str(epoch) + '/' + str(epoches))
                # training by time
                train_loss, train_er = self.run_one_epoch(
                    data_set=self._train_set,
                    batch_size=mini_batch_size,
                    is_train=True
                )
                # valid by time
                valid_loss, valid_er = self.run_one_epoch(
                    data_set=self._valid_set,
                    batch_size=valid_batch_size,
                    is_train=False
                )

                epoch_list.append(epoch)
                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)
                train_er_list.append(train_er)
                valid_er_list.append(valid_er)
                console.log('info', 'Train Loss', str(train_loss) + '\n')
                console.log('info', 'Valid Loss', str(valid_loss) + '\n')
                console.log('info', 'Train ER', str(train_er) + '\n')
                console.log('info', 'Valid ER', str(valid_er) + '\n')

                if True:
                    fig = plt.figure(figsize=(12, 12))
                    cost_plt = fig.add_subplot(211)
                    er_plt = fig.add_subplot(212)
                    cost_plt.title.set_text('Loss')
                    er_plt.title.set_text('Error Rate')
                    cost_plt.plot(
                        epoch_list, train_loss_list, 'g',
                        epoch_list, valid_loss_list, 'r'
                    )
                    er_plt.plot(
                        epoch_list, train_er_list, 'g',
                        epoch_list, valid_er_list, 'r'
                    )
                    plt.savefig(
                        os.path.join(self._path, 'error.png')
                    )
                    plt.clf()
                    plt.close(fig)

                if valid_loss < best_valid_loss:
                    if self.checkpoint_epoch(epoch, epoches) or\
                       epoch > 400:
                        best_valid_loss = valid_loss
                        self._model.save(self._sess)
                    early_stopping_cnt = 0
                else:
                    early_stopping_cnt += 1
                    # if early_stopping_n > 0 && early_s

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
        data_set.batch_size = batch_size
        total_samples = data_set.length
        batches = int((total_samples - 1) / batch_size) + 1

        batch = 0
        avg_er = 0
        avg_loss = 0
        avg_count = 0
        data_set.reset()
        while True:
            feed_dict, sub_set = self.next_feed_dict(data_set)
            if feed_dict is None:
                break
            to_run = [self._model.loss_fn, self._model._silence_pred]
            if is_train:
                to_run.append(self._optimizer)
            result = self._sess.run(
                to_run,
                feed_dict=feed_dict
            )
            # print information
            bar = process_bar.process_bar(batch, batches)
            batch += 1  # add one batch
            avg_count += 1
            avg_loss += result[0]
            loss_str = '\tLoss %.4f' % (avg_loss / avg_count)
            try:
                avg_er += self._model.error_rate(
                    pred=result[1],
                    true=sub_set[self.label_key]
                )
            except:
                pass

            loss_str += '\tER %.4f' % (avg_er / avg_count)
            if is_train:
                console.log('', 'Train', bar + loss_str + '\r')
            else:
                console.log('', 'Valid', bar + loss_str + '\r')
        console.log()
        return avg_loss / avg_count, avg_er / avg_count
