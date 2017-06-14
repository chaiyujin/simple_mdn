import os
import pickle
import tensorflow as tf
import numpy as np
from pack.models.gan2 import GAN
from pack.backend.data import data_loader


if __name__ == '__main__':
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

    scale_times = 4

    config = {
        'b_size': 1,
        'img_h': 75 * scale_times,
        'img_w': 16,
        'z_dims': 16,
        'expr': 19,
        'penalty_scale': 10,
        'l1_scale': 3
    }

    def scale_sample(net, sess, set, skip, save_path, noise):
        def concat(x):
            list = []
            for i in range(x.shape[0]):
                list.append(x[i])
            x = np.concatenate(list, axis=0)
            return np.asarray([x])

        set.batch_size = scale_times
        set.reset()
        i = 0
        while True:
            for _ in range(skip):
                vb, bs = set.next_batch()
                if vb is None:
                    break
            if vb is None or bs != scale_times:
                break
            vx = vb['inputs']
            vy = vb['outputs']
            pf = vb['path_prefix']
            vz = np.asarray(noise) if noise is not None else vb['noise']
            xx = concat(vx)
            yy = concat(vy)
            zz = concat(vz)
            files = []
            for file in pf:
                files.append(file)
            net.scale_sample(sess, save_path, i, xx, yy, zz, files)
            # break
            i += 1

    net = GAN(config=config)
    with tf.Session() as sess:
        net.saver.restore(sess, 'save/best.cpkt')
        scale_sample(net, sess, train_set, 200, 'sample_scale_train', None)
        train_set.reset()
        batch, _ = train_set.next_batch()
        noise = batch['noise']
        scale_sample(net, sess, no_train, 100, 'sample_scale_unseen', noise)
