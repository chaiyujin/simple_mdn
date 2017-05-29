from backend.data import data_process, data_loader
from model.cross_entropy import model
from backend.train import BPTT
import sys
import tensorflow as tf
from backend.utils import console

if __name__ == '__main__':
    sys.path.append('.')
    config = {
        'audio_num_features': 39,
        'anime_num_features': 19,
        'audio_lstm_size': 128,
        'anime_lstm_size': 256,
        'audio_lstm_dropout': 0,
        'anime_lstm_dropout': 0,
        'phoneme_classes': 20,
        'dense_size0': 128,
        'dense_size1': 64,
        'train': 0,
        'dropout': 0
    }
    data_process.process(config)
    # set model
    model = model.Model(config)
    # load data set
    set_keys = ['inputs', 'outputs', 'seq_len', 'path_prefix']
    train_set = data_loader.DataSet(keys=set_keys)
    valid_set = data_loader.DataSet(keys=set_keys)
    train_set.add_pkl('data/train.pkl')
    valid_set.add_pkl('data/test.pkl')
    mean, stdv = train_set.normalize('inputs')
    valid_set.normalize('inputs', mean, stdv)
    train_set.power('outputs', 0.25)
    valid_set.power('outputs', 0.25)

    with tf.Session() as sess:
        if model._train:
            console.add_log_file('log.txt')
            trainer = BPTT.Trainer(
                model, train_set, valid_set, label_key='outputs',
                feed_keys={
                    'inputs': 'inputs',
                    'outputs': 'outputs',
                    'seq_len': 'seq_len'
                }
            )
            optimizer = tf.train.AdamOptimizer(1e-4)
            trainer.train(
                sess,
                optimizer,
                5000,
                8,
                64,
                early_stopping_n=3,
                load=False
            )
            optimizer = tf.train.AdamOptimizer(1e-6)
            trainer.train(
                sess,
                optimizer,
                5000,
                8,
                64,
                early_stopping_n=5,
                load=True
            )
            console.close_log_files()
        else:
            model.load(sess)
            model.sample(sess, valid_set, 8, 8, True)


