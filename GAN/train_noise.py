import os
import argparse
import numpy as np
import tensorflow as tf
from pack.models.gan2_train_noise import GAN
from pack.backend.data import data_loader


if __name__ == '__main__':
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
        for i in range(18, 19):
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
    with tf.Session() as sess:
        if args.train:
            sess.run(tf.global_variables_initializer())
            net.saver.restore(sess, 'save/best.cpkt')
            train_set.batch_size = 32
            train_set.reset()
            valid_set.batch_size = 32
            valid_set.reset()
            vb, _ = train_set.next_batch()
            vx = vb['inputs']
            vy = vb['outputs']
            vz = None
            pf = vb['path_prefix']

            train_set.reset()
            for epoch in range(10000):
                batch, bs = train_set.next_batch()
                if batch is None or bs != 32:
                    train_set.reset()
                    batch, bs = train_set.next_batch()
                xb = batch['inputs']
                yb = batch['outputs']
                zb = None

                net.train_batch(sess, epoch, xb, yb, zb, vx, vy, vz, pf)
        else:
            net.saver.restore(sess, 'save_noise/model.cpkt')
            net.noise_saver.restore(sess, 'save_noise/best.cpkt')
            train_set.batch_size = 32
            train_set.reset()
            i = 0
            while True:
                for _ in range(1):
                    vb, bs = train_set.next_batch()
                    if vb is None:
                        break
                if vb is None or bs != 32:
                    break
                vx = vb['inputs']
                vy = vb['outputs']
                vz = None
                pf = vb['path_prefix']
                net.sample(sess, 'sample_noise_train', i, vx, vy, vz, pf)
                i += 1
