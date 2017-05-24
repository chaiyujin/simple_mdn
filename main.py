import sys
import numpy
import data_process
import tensorflow as tf
from ce_model import Model
from utils.process_bar import process_bar
from utils import console
from train import RTRL


if __name__ == '__main__':
    config = {
        'audio_num_features': 39,
        'anime_num_features': 19,
        'audio_lstm_size': 128,
        'anime_lstm_size': 50,
        'dense_size': 400,
        'mdn_K': 5,
        'mdn_bias': 8,
        'phoneme_classes': 40,
        'train': 1,
        'dropout': 0.5
    }

    model = Model(config)

    data_process.process(config)
    train_data = data_process.load('data/train.pkl')
    valid_data = data_process.load('data/test.pkl')
    train_data['outputs'] = train_data['outputs'] ** 0.25
    valid_data['outputs'] = train_data['outputs'] ** 0.25
    mean = numpy.mean(train_data['inputs'])
    stdv = numpy.std(train_data['inputs'])
    train_data['inputs'] = (train_data['inputs'] - mean) / stdv
    valid_data['inputs'] = (valid_data['inputs'] - mean) / stdv
    print(numpy.amin(train_data['outputs']), ' ',
          numpy.amax(train_data['outputs']))
    print(mean, stdv)

    if model._train:
        # pass
        console.add_log_file('log.txt')
        optimizer = tf.train.AdamOptimizer(1e-5)
        # optimizer = tf.train.MomentumOptimizer(1e-4, 0.9)
        model.simple_train(
            train_data=train_data,
            valid_data=valid_data,
            epoches=5000,
            mini_batch_size=16,
            valid_batch_size=64,
            optimizer=optimizer
        )
        console.close_log_files()
        # optimizer = tf.train.GradientDescentOptimizer(1e-4)
        # trainer = RTRL.Trainer(model, train_data, valid_data)
        # trainer.train(
        #     optimizer,
        #     5000,
        #     64,
        #     64
        # )

    with tf.Session() as sess:
        model.load(sess)
        print(model.sample_data(sess, train_data, 8, 8, True))
        # print(model.sample_data(sess, valid_data, 8, 8, True))
