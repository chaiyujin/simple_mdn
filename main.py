import numpy
import data_process
import tensorflow as tf
from model import Model
from utils.process_bar import process_bar
from utils import console


if __name__ == '__main__':
    config = {
        'audio_num_features': 39,
        'anime_num_features': 19,
        'audio_lstm_size': 128,
        'anime_lstm_size': 50,
        'dense_size': 400,
        'mdn_K': 5,
        'mdn_bias': 5,
        'phoneme_classes': 40,
        'train': 1,
        'dropout': 0.5
    }

    model = Model(config)

    data_process.process(config)
    train_data = data_process.load('data/train.pkl')
    valid_data = data_process.load('data/test.pkl')
    mean = numpy.mean(train_data['inputs'])
    stdv = numpy.std(train_data['inputs'])
    train_data['inputs'] = (train_data['inputs'] - mean) / stdv
    valid_data['inputs'] = (valid_data['inputs'] - mean) / stdv
    print(numpy.amin(train_data['outputs']), ' ',
          numpy.amax(train_data['outputs']))
    print(mean, stdv)

    if model._train:
        optimizer = tf.train.MomentumOptimizer(1e-3, 0.9)
        model.simple_train(
            train_data=train_data,
            valid_data=valid_data,
            epoches=10000,
            mini_batch_size=4,
            valid_batch_size=32,
            optimizer=optimizer
        )

    with tf.Session() as sess:
        model.load(sess)
        print(model.sample_data(sess, valid_data, 8, 8, True))
