from backend.data import data_process, data_loader
from model.silence import model
from backend.train import BPTT
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sys.path.append('.')
    config = {
        'audio_num_features': 39,
        'anime_num_features': 19,
        'audio_lstm_size': 128,
        'audio_lstm_dropout': 0.5,
        'phoneme_classes': 40,
        'dense_size': 100,
        'train': 1,
        'dropout': 0.5
    }
    data_process.process(config)
    # set model
    model = model.Model(config)
    # load data set
    train_set = data_loader.DataSet(keys=['inputs', 'silence', 'seq_len'])
    valid_set = data_loader.DataSet(keys=['inputs', 'silence', 'seq_len'])
    train_set.add_pkl('data/train.pkl')
    valid_set.add_pkl('data/test.pkl')
    train_set.normalize('inputs')
    valid_set.normalize('inputs')

    with tf.Session() as sess:
        if model._train:
            optimizer = tf.train.GradientDescentOptimizer(1e-2)
            trainer = BPTT.Trainer(
                model, train_set, valid_set, label_key='silence',
                feed_keys={
                    'inputs': 'inputs',
                    'outputs': 'silence',
                    'seq_len': 'seq_len'
                }
            )
            trainer.train(
                sess,
                optimizer,
                5000,
                64,
                64,
                load=False
            )



