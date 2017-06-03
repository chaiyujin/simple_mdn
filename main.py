import os
import sys
import argparse
import tensorflow as tf
from model.cross_entropy import model
from backend.train import BPTT
from backend.utils import console
from backend.data import data_loader

if __name__ == '__main__':
    sys.path.append('.')

    # parse the command
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--process_data', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    args = parser.parse_args()

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
        'train': args.train,
        'dropout': 0
    }

    if args.process_data:
        from backend.data import data_process
        train_list = []
        test_list = []
        for i in range(1, 2):
            path = '../../dataset/GRID/video/s' + str(i) + '/'
            if not os.path.exists(path):
                continue
            save_train = 'data/train' + str(i) + '.pkl'
            save_test = 'data/test' + str(i) + '.pkl'
            train_list.append(save_train)
            test_list.append(save_test)
            data_process.process(
                config, path, 'mpg', 0.2, save_train, save_test)
        data_process.merge(train_list, 'data/train.pkl')
        data_process.merge(test_list, 'data/test.pkl')
    # set model
    model = model.Model(config)
    # load data set
    set_keys = ['inputs', 'outputs', 'seq_len', 'path_prefix']
    train_set = data_loader.DataSet(keys=set_keys)
    valid_set = data_loader.DataSet(keys=set_keys)
    save_train = 'data/train.pkl'
    save_test = 'data/test.pkl'
    train_set.add_pkl(save_train)
    valid_set.add_pkl(save_test)
    print('Train: ', train_set.length)
    print('Valid: ', valid_set.length)
    mean, stdv = train_set.normalize('inputs')
    print('mean: ', mean, '\tstdv: ', stdv)
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
                train_id=0,
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
                train_id=1,
                early_stopping_n=5,
                load=True
            )
            console.close_log_files()
        else:
            model.load(sess)
            model.sample(sess, valid_set, 8, 8, True)


