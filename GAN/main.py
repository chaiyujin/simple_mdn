import os
import sys
import pickle
import argparse
import tensorflow as tf
from pack.models.gan2 import GAN, generate_noise
from pack.backend.data import data_loader


def add_noise(save_train, save_test=None):
    train_set = None
    test_set = None
    len_train = 0
    len_test = 0
    with open(save_train, 'rb') as file:
        train_set = pickle.load(file)
        len_train = len(train_set['inputs'])
    if save_test:
        with open(save_test, 'rb') as file:
            test_set = pickle.load(file)
        len_test = len(test_set['inputs'])
    noise_config = {
        'length': len_train + len_test,
        'time_steps': 75,
        'chr_dims': 16,
        'emo_dims': 12,
        'rdm_dims': 4
    }
    noise = generate_noise(noise_config)
    print('Add noise with shape: ', noise.shape)
    # train
    train_noise = noise[:len_train]
    train_set['noise'] = train_noise
    print(len_train, ' ', train_noise.shape)
    with open(save_train, 'wb') as file:
        pickle.dump(train_set, file)

    if save_test:
        test_noise = noise[len_train:]
        test_set['noise'] = test_noise
        print(len_test, ' ', test_noise.shape)

        with open(save_test, 'wb') as file:
            pickle.dump(test_set, file)


if __name__ == '__main__':
    sys.path.append('.')

    # parse the command
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--process_data', action='store_true')
    # parser.add_argument('-m', '--merge_data', action='store_true')
    parser.add_argument('-n', '--noise_data', action='store_true')
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
            if args.noise_data:
                add_noise(save_train, save_test)
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
            if args.noise_data:
                add_noise(save_train)
        data_process.merge(train_list, 'data/notrain.pkl')
    # load data set
    set_keys = ['inputs', 'outputs', 'noise', 'path_prefix']
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
            vz = vb['noise']
            pf = vb['path_prefix']

            for epoch in range(1000000):
                batch, bs = train_set.next_batch()
                if batch is None or bs != 32:
                    train_set.reset()
                    batch, bs = train_set.next_batch()
                xb = batch['inputs']
                yb = batch['outputs']
                zb = batch['noise']

                net.train_batch(sess, epoch, xb, yb, zb, vx, vy, vz, pf)
        else:
            net.saver.restore(sess, 'save/best.cpkt')

            # 1. sample train
            no_train.batch_size = 32
            no_train.reset()
            train_set.batch_size = 32
            train_set.reset()
            i = 0
            while True:
                for _ in range(25):
                    vb, bs = train_set.next_batch()
                    if vb is None:
                        break
                if vb is None or bs != 32:
                    break
                vx = vb['inputs']
                vy = vb['outputs']
                pf = vb['path_prefix']
                vz = vb['noise']
                print(vz[0][0])
                net.sample(sess, 'sample_train', i, vx, vy, vz, pf)
                i += 1

            # 1. sample train
            valid_set.batch_size = 32
            valid_set.reset()
            i = 0
            while True:
                for _ in range(5):
                    vb, bs = valid_set.next_batch()
                    if vb is None:
                        break
                if vb is None or bs != 32:
                    break
                vx = vb['inputs']
                vy = vb['outputs']
                pf = vb['path_prefix']
                vz = vb['noise']
                print(vz[0][0])
                net.sample(sess, 'sample_valid', i, vx, vy, vz, pf)
                i += 1

            # 2. sample unseen with given noise
            no_train.batch_size = 32
            no_train.reset()
            train_set.batch_size = 32
            train_set.reset()
            vb, bs = train_set.next_batch()
            vz = vb['noise']
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
                print(vz[0][0])
                net.sample(sess, 'sample_unseen_with_given_noise', i, vx, vy, vz, pf)
                i += 1

            # 3. sample with random noise
            no_train.batch_size = 32
            no_train.reset()
            train_set.batch_size = 32
            train_set.reset()
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
                vz = vb['noise']
                print(vz[0][0])
                net.sample(sess, 'sample_unseen_with_random_noise', i, vx, vy, vz, pf)
                i += 1

            # 4. sample with sample_different_noise
            no_train.batch_size = 32
            no_train.reset()
            train_set.batch_size = 32
            train_set.reset()
            for _ in range(5 * 20):
                vb, bs = no_train.next_batch()
            vx = vb['inputs']
            vy = vb['outputs']
            pf = vb['path_prefix']
            i = 0
            while True:
                for _ in range(25):
                    vb, bs = train_set.next_batch()
                    if vb is None:
                        break
                if vb is None or bs != 32:
                    break
                vz = vb['noise']
                print(vz[0][0])
                net.sample(sess, 'sample_different_noise', i, vx, vy, vz, pf)
                i += 1
