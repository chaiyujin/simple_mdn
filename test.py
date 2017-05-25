from backend.data import data_process, data_loader
from model.cross_entropy import model
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
    train_set = data_loader.DataSet(keys=['inputs', 'outputs', 'seq_len', 'path_prefix'])
    valid_set = data_loader.DataSet(keys=['inputs', 'outputs', 'seq_len', 'path_prefix'])
    train_set.add_pkl('data/train.pkl')
    valid_set.add_pkl('data/test.pkl')
    train_set.normalize('inputs')
    valid_set.normalize('inputs')
    train_set.power('outputs', 0.25)
    valid_set.power('outputs', 0.25)

    with tf.Session() as sess:
        if model._train:
            optimizer = tf.train.GradientDescentOptimizer(1e-2)
            trainer = BPTT.Trainer(
                model, train_set, valid_set, label_key='outputs',
                feed_keys={
                    'inputs': 'inputs',
                    'outputs': 'outputs',
                    'seq_len': 'seq_len'
                }
            )
            trainer.train(
                sess,
                optimizer,
                5000,
                64,
                64,
                early_stopping_n=10,
                load=False
            )

        model.load(sess)

        model.sample(sess, valid_set, 8, 8, True)


