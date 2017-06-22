import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
from pack.models.gan_ import GAN
from pack.backend.data import data_loader


def get_trained_character_noise():
    with open('data/all_char.pkl', 'rb') as file:
        return pickle.load(file)


if __name__ == '__main__':
    id = 18
    config = {
        'audio_num_features': 39,
        'anime_num_features': 19,
    }
    # parse the command
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--process_data', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    args = parser.parse_args()
    if args.process_data:
        from pack.backend.data import data_process
        train_list = []
        test_list = []
        for i in range(id, id + 1):
            path = '../../../dataset/GRID/video/s' + str(i) + '/'
            if not os.path.exists(path):
                continue
            save_train = 'data/noise_train' + str(i) + '.pkl'
            save_test = 'data/noise_test' + str(i) + '.pkl'
            train_list.append(save_train)
            test_list.append(save_test)
            data_process.process(
                config, path, 'mpg', 0.2, save_train, save_test)
        data_process.merge(train_list, 'data/noise_train.pkl')
        data_process.merge(test_list, 'data/noise_test.pkl')

    set_keys = ['inputs', 'outputs', 'path_prefix']
    train_set = data_loader.DataSet(keys=set_keys)
    valid_set = data_loader.DataSet(keys=set_keys)
    save_train = 'data/noise_train.pkl'
    save_test = 'data/noise_test.pkl'
    train_set.add_pkl(save_train)
    valid_set.add_pkl(save_test)
    print('Train: ', train_set.length)
    print('Valid: ', valid_set.length)
    mean, stdv = 1.26696129707, 22.317331374549003
    print('mean: ', mean, '\tstdv: ', stdv)
    train_set.normalize('inputs', mean, stdv)
    valid_set.normalize('inputs', mean, stdv)

    net = GAN()
    char_noise = get_trained_character_noise()
    with tf.Session() as sess:
        net.saver.restore(sess, 'save/best.cpkt')
        if args.train:
            train_set.batch_size = 32
            train_set.reset()
            valid_set.batch_size = 32
            valid_set.reset()
            for _ in range(20):
                vb, _ = train_set.next_batch()
            vx = vb['inputs']
            vy = vb['outputs']
            vz = char_noise[12]
            pf = vb['path_prefix']

            train_set.reset()
            for epoch in range(10000):
                batch, bs = train_set.next_batch()
                if batch is None or bs != 32:
                    train_set.reset()
                    batch, bs = train_set.next_batch()
                xb = batch['inputs']
                yb = batch['outputs']
                zb = vz

                vz = net.train_batch(
                    sess, epoch, id, xb, yb, zb, vx, vy, vz, pf, True
                )
                # if epoch % 10 == 0:
                #     print(vz)
                if epoch % 100 == 0:
                    with open('save/noise.pkl', 'wb') as file:
                        pickle.dump(vz, file)
        else:
            valid_set.batch_size = 32
            valid_set.reset()
            i = 0
            vz = None
            with open('save/noise.pkl', 'rb') as file:
                vz = pickle.load(file)
            while True:
                for _ in range(1):
                    vb, bs = valid_set.next_batch()
                    if vb is None:
                        break
                if vb is None or bs != 32:
                    break
                vx = vb['inputs']
                vy = vb['outputs']
                pf = vb['path_prefix']
                net.sample(sess, 'sample_noise_valid', i, vx, vy, vz, pf)
                i += 1
