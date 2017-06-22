import os
import sys
import pickle
import argparse
import numpy as np
import tensorflow as tf
from pack.models.gan_ import GAN
from pack.models.gan2 import generate_noise
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


def make_character_noise():
    def sample_z(n, l=-1., r=1.):
        return np.random.uniform(l, r, size=[n])

    for i in range(1, 34):
        z = sample_z(16)
        with open('data/char' + str(i) + '.pkl', 'wb') as file:
            pickle.dump(z, file)


def load_character_noise():
    noise = {}
    for i in range(1, 34):
        with open('data/char' + str(i) + '.pkl', 'rb') as file:
            noise[i] = pickle.load(file)
    return noise


def save_character_noise(noise):
    with open('data/all_char.pkl', 'wb') as file:
        pickle.dump(noise, file)


def get_trained_character_noise():
    with open('data/all_char.pkl', 'rb') as file:
        return pickle.load(file)


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

    bad = [4, 15, 16]
    fair = [9, 13, 14]
    if args.noise_data:
        make_character_noise()
    if args.process_data:
        from pack.backend.data import data_process
        train_list = []
        test_list = []
        for i in range(1, 17):
            if i in bad or i in fair:
                continue
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
    mean, stdv = 1.4299498196, 22.017510094309422
    set_keys = ['inputs', 'outputs', 'noise', 'path_prefix']
    valid_set = data_loader.DataSet(keys=set_keys)
    no_train = data_loader.DataSet(keys=set_keys)
    save_test = 'data/test.pkl'
    save_no = 'data/notrain.pkl'
    valid_set.add_pkl(save_test)
    no_train.add_pkl(save_no)
    print('Valid: ', valid_set.length)
    print('No Train: ', no_train.length)
    print('mean: ', mean, '\tstdv: ', stdv)
    valid_set.normalize('inputs', mean, stdv)
    no_train.normalize('inputs', mean, stdv)

    batch_size = 32
    train_sets = {}
    valid_sets = {}
    for i in range(1, 17):
        save_train = 'data/train' + str(i) + '.pkl'
        save_valid = 'data/test' + str(i) + '.pkl'
        if not os.path.exists(save_train):
            train_sets[i] = None
            valid_sets[i] = None
            continue
        train_sets[i] = data_loader.DataSet(keys=set_keys)
        train_sets[i].add_pkl(save_train)
        train_sets[i].normalize('inputs', mean, stdv)
        train_sets[i].batch_size = batch_size
        train_sets[i].reset()

        valid_sets[i] = data_loader.DataSet(keys=set_keys)
        valid_sets[i].add_pkl(save_valid)
        valid_sets[i].normalize('inputs', mean, stdv)
        valid_sets[i].batch_size = batch_size
        valid_sets[i].reset()

    net = GAN()
    with tf.Session() as sess:
        if args.train:
            char_noise = load_character_noise()
            sess.run(tf.global_variables_initializer())
            valid_set.batch_size = batch_size
            valid_set.reset()
            for _ in range(5):
                vb, _ = valid_set.next_batch()
            vx = vb['inputs']
            vy = vb['outputs']
            vz = vb['noise']
            pf = vb['path_prefix']

            i = 1
            train_sets[i].reset()
            for epoch in range(1000000):
                batch, bs = train_sets[i].next_batch()
                if batch is None or bs != batch_size:
                    i += 1
                    if (i >= 17):
                        i = 1
                    while (train_sets[i] is None):
                        i += 1
                        if (i >= 17):
                            i = 1
                    train_sets[i].reset()
                    batch, bs = train_sets[i].next_batch()
                xb = batch['inputs']
                yb = batch['outputs']
                zb = char_noise[i]
                vz = char_noise[1]

                new_z = net.train_batch(
                    sess, epoch, i, xb, yb, zb, vx, vy, vz, pf
                )
                char_noise[i] = new_z

                if epoch % 10 == 0:
                    print()
                    for j in range(1, 17, 4):
                        info = ''
                        for k in range(4):
                            info += str(j + k) + ':' +\
                                    '%.6f' % char_noise[j + k].mean() +\
                                    '\t'
                        print(info)
                    if epoch % 100 == 0:
                        save_character_noise(char_noise)
        else:
            char_noise = get_trained_character_noise()
            net.saver.restore(sess, 'save/best.cpkt')
            # 1. sample train
            no_train.batch_size = batch_size
            no_train.reset()
            for idx in range(1, 17):
                if train_sets[idx] is None:
                    continue
                train_sets[idx].batch_size = batch_size
                train_sets[idx].reset()
                i = 0
                for j in range(800):
                    vb, bs = train_sets[idx].next_batch()
                    if vb is None or bs != batch_size:
                        break
                    if j % 10 == 0:
                        vx = vb['inputs']
                        vy = vb['outputs']
                        pf = vb['path_prefix']
                        vz = char_noise[idx]
                        net.sample(
                            sess, 'sample_train',
                            str(idx) + '_' + str(i),
                            vx, vy, vz, pf
                        )
                        i += 1

            # 2. sample unseen with given noise
            no_train.batch_size = 32
            no_train.reset()
            vz = char_noise[1]
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
                net.sample(
                    sess, 'sample_unseen_with_given_noise',
                    i, vx, vy, vz, pf
                )
                i += 1

            # 4. sample with sample_different_noise
            no_train.batch_size = 32
            no_train.reset()
            for _ in range(5 * 20):
                vb, bs = no_train.next_batch()
            vx = vb['inputs']
            vy = vb['outputs']
            pf = vb['path_prefix']
            for idx in range(1, 17):
                if train_sets[idx] is None:
                    continue
                vz = char_noise[idx]
                net.sample(
                    sess, 'sample_different_noise',
                    idx, vx, vy, vz, pf
                )

            for idx in range(1, 17):
                if valid_sets[idx] is None:
                    continue
                valid_sets[idx].batch_size = batch_size
                valid_sets[idx].reset()
                i = 0
                for j in range(800):
                    vb, bs = valid_sets[idx].next_batch()
                    if vb is None or bs != batch_size:
                        break
                    if j % 3 == 0:
                        vx = vb['inputs']
                        vy = vb['outputs']
                        pf = vb['path_prefix']
                        vz = char_noise[idx]
                        net.sample(
                            sess, 'sample_valid',
                            str(idx) + '_' + str(i),
                            vx, vy, vz, pf
                        )
                        i += 1
