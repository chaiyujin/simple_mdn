import os
import sys
import argparse
import tensorflow as tf
from pack.models.gan import GAN
from pack.backend.data import data_loader

if __name__ == '__main__':
    sys.path.append('.')

    # parse the command
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--process_data', action='store_true')
    parser.add_argument('-m', '--merge_data', action='store_true')
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
        from pack.backend.data import data_process
        train_list = []
        test_list = []
        for i in range(1, 17):
            path = '../../../dataset/GRID/video/s' + str(i) + '/'
            if not os.path.exists(path):
                continue
            save_train = 'data/train' + str(i) + '.pkl'
            save_test = 'data/test' + str(i) + '.pkl'
            train_list.append(save_train)
            test_list.append(save_test)
            data_process.process(
                config, path, 'mpg', 0.2, save_train, save_test)
        if args.merge_data:
            data_process.merge(train_list, 'data/train.pkl')
            data_process.merge(test_list, 'data/test.pkl')

        train_list = []
        for i in range(18, 32):
            path = '../../../dataset/GRID/video/s' + str(i) + '/'
            if not os.path.exists(path):
                continue
            save_train = 'data/train' + str(i) + '.pkl'
            save_test = 'data/test' + str(i) + '.pkl'
            train_list.append(save_train)
            test_list.append(save_test)
            data_process.process(
                config, path, 'mpg', 0, save_train, save_test)
        if args.merge_data:
            data_process.merge(train_list, 'data/notrain.pkl')
    # load data set
    set_keys = ['inputs', 'outputs', 'seq_len', 'path_prefix']
    train_set = data_loader.DataSet(keys=set_keys)
    valid_set = data_loader.DataSet(keys=set_keys)
    no_train = data_loader.DataSet(keys=set_keys)
    save_train = 'data/train.pkl'
    save_test = 'data/test.pkl'
    save_no = 'data/notrain.pkl'
    train_set.add_pkl(save_train)
    valid_set.add_pkl(save_test)
    no_train.add_pkl(save_no)
    print('Train: ', train_set.length)
    print('Valid: ', valid_set.length)
    print('No Train: ', no_train.length)
    mean, stdv = train_set.normalize('inputs')
    print('mean: ', mean, '\tstdv: ', stdv)
    valid_set.normalize('inputs', mean, stdv)
    no_train.normalize('inputs', mean, stdv)

    net = GAN()

    with tf.Session() as sess:
        if args.train:
            sess.run(tf.global_variables_initializer())
            train_set.batch_size = 32
            train_set.reset()
            valid_set.batch_size = 32
            valid_set.reset()
            vb, _ = valid_set.next_batch()
            vx = vb['inputs']
            vy = vb['outputs']
            pf = vb['path_prefix']

            for epoch in range(1000000):
                batch, bs = train_set.next_batch()
                if batch is None or bs != 32:
                    train_set.reset()
                    batch, bs = train_set.next_batch()
                xb = batch['inputs']
                yb = batch['outputs']

                net.train_batch(sess, epoch, xb, yb, vx, vy, pf)
        else:
            net.saver.restore(sess, 'save/best.cpkt')
            no_train.batch_size = 32
            no_train.reset()
            i = 0
            while True:
                for _ in range(5):
                    vb, bs = no_train.next_batch()
                    if vb is None:
                        break
                if vb is None or bs != 32:
                    break
                vx = vb['inputs']
                vy = vb['outputs']
                pf = vb['path_prefix']
                net.sample(sess, 'sample', i, vx, vy, pf)
                i += 1
