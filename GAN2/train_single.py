import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.layers as tflayers
from pack.backend.data import data_loader
from pack.models.models import ConvNet, DeconvNet


def generator(x, scope='gen', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        x_s = [int(d) for d in x.get_shape()]
        enc = ConvNet(config={'log'})
        dec = DeconvNet(config={'log'})
        # input 75x16, 1
        enc.conv_layer(16, [1, 4], [1, 2], 'lrelu')
        enc.conv_layer(64, [1, 4], [1, 2], 'lrelu', True)
        # enc.conv_layer(64, [1, 4], [1, 2], 'lrelu', True)
        # dec.deconv_layer(64, [2, 4], [1, 2], 'lrelu', True)
        dec.deconv_layer(32, [3, 4], [1, 2], 'lrelu', True)
        dec.deconv_layer(16, [2, 4], [1, 2], 'lrelu', True)
        lin = tf.reshape(dec(enc(x)), [x_s[0] * x_s[1], -1])
        logits = tflayers.fully_connected(
            lin, 19, None,
            weights_initializer=tf.random_normal_initializer(0, 0.02)
        )
        logits = tf.reshape(
            logits, [x_s[0], x_s[1], 19, 1]
        )
        anime = tf.sigmoid(logits)

        return anime, logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    args = parser.parse_args()

    mean, stdv = 1.4299498196, 22.017510094309422
    set_keys = ['inputs', 'outputs', 'noise', 'path_prefix']

    # net
    batch_size = 32
    x_s = [batch_size, 75, 16, 1]
    y_s = [batch_size, 75, 19, 1]
    x = tf.placeholder(tf.float32, x_s)
    y = tf.placeholder(tf.float32, y_s)
    y_fake, y_fake_logits = generator(x)
    loss_fn = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y, logits=y_fake_logits
    )
    optim = tf.train.AdamOptimizer(1e-4).minimize(loss_fn)
    saver = tf.train.Saver()
    if args.train:
        train_set = data_loader.DataSet(keys=set_keys)
        valid_set = data_loader.DataSet(keys=set_keys)
        save_train = 'data/train1.pkl'
        save_valid = 'data/test1.pkl'
        train_set.add_pkl(save_train)
        valid_set.add_pkl(save_valid)
        train_set.normalize('inputs', mean, stdv)
        valid_set.normalize('inputs', mean, stdv)
        train_set.batch_size = batch_size
        valid_set.batch_size = batch_size
        train_set.reset()
        valid_set.reset()

        # train
        epoch_list = []
        train_losses = []
        valid_losses = []
        if not os.path.exists('save_single'):
            os.mkdir('save_single')
        if not os.path.exists('out'):
            os.mkdir('out')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # get a sample
            for _ in range(5):
                vb, _ = valid_set.next_batch()
            vx, vy, pf = vb['inputs'], vb['outputs'], vb['path_prefix']
            vx = np.reshape(vx, x_s)
            vy = np.reshape(vy, y_s)

            n_early_stop = 10
            best_valid_loss = 1000000
            for epoch in range(10000):
                train_set.reset()
                valid_set.reset()
                # train all train set data
                train_loss = 0
                train_cnt = 0
                while True:
                    batch, bs = train_set.next_batch()
                    if batch is None or bs != batch_size:
                        break
                    xb, yb = batch['inputs'], batch['outputs']
                    xb = np.reshape(xb, x_s)
                    yb = np.reshape(yb, y_s)
                    loss, _ = sess.run(
                        [loss_fn, optim],
                        feed_dict={x: xb, y: yb}
                    )
                    train_loss += loss.mean()
                    train_cnt += 1
                # valid phase
                valid_loss = 0
                valid_cnt = 0
                while True:
                    batch, bs = valid_set.next_batch()
                    if batch is None or bs != batch_size:
                        break
                    xb, yb = batch['inputs'], batch['outputs']
                    xb = np.reshape(xb, x_s)
                    yb = np.reshape(yb, y_s)
                    loss = sess.run([loss_fn], feed_dict={x: xb, y: yb})[0]
                    valid_loss += loss.mean()
                    valid_cnt += 1
                # sample
                if epoch % 20 == 0:
                    pred = sess.run(y_fake, feed_dict={x: vx})
                    from pack.backend.utils.media import sample_video
                    sample_config = {
                        'path_prefix': pf[0],
                        'anime_pred': pred[0]
                    }
                    if epoch == 0:
                        sample_config['anime_true'] = vy[0]
                    sample_video(
                        sample_config,
                        os.path.join('out', str(epoch) + '.mp4'))
                # print info
                train_loss /= train_cnt
                valid_loss /= valid_cnt
                epoch_list.append(epoch)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                # save
                if valid_loss < best_valid_loss:
                    n_early_stop = 10
                    best_valid_loss = valid_loss
                    saver.save(sess, 'save_single/best.ckpt')
                elif valid_loss > best_valid_loss:
                    n_early_stop -= 1
                    if n_early_stop == 0:
                        break
                # draw
                if epoch % 10 == 0:
                    fig = plt.figure(figsize=(12, 12))
                    loss_plt = fig.add_subplot(111)
                    loss_plt.title.set_text('Loss')
                    loss_plt.plot(
                        epoch_list, train_losses, 'g',
                        epoch_list, valid_losses, 'r'
                    )
                    plt.savefig('error.png')
                    plt.clf()
                    plt.close(fig)
                # info
                info = 'Epoch ' + str(epoch) + ' '
                info += 'Train_loss: ' + '%.4f' % train_loss + ' '
                info += 'Valid_loss: ' + '%.4f' % valid_loss + ' '
                info += 'Early_stop: ' + str(n_early_stop)
                print(info)
    else:
        valid_sets = {}
        for i in range(1, 34):
            save_path = 'data/train' + str(i) + '.pkl'
            if not os.path.exists(save_path):
                continue
            valid_set = data_loader.DataSet(keys=set_keys)
            valid_set.add_pkl(save_path)
            valid_set.normalize('inputs', mean, stdv)
            valid_set.batch_size = batch_size
            valid_sets[i] = valid_set

        with tf.Session() as sess:
            saver.restore(sess, 'save_single/best.ckpt')
            if not os.path.exists('sample_single'):
                os.mkdir('sample_single')
            for i in valid_sets:
                valid_set = valid_sets[i]
                for j in range(800):
                    batch, bs = valid_set.next_batch()
                    if batch is None or bs != batch_size:
                        break
                    xb, yb = batch['inputs'], batch['outputs']
                    xb = np.reshape(xb, x_s)
                    yb = np.reshape(yb, y_s)
                    pf = batch['path_prefix']
                    if j % 5 == 0:
                        idx = str(i) + '_' + str(int(j/5))
                        pred = sess.run(y_fake, feed_dict={x: xb})
                        from pack.backend.utils.media import sample_video
                        sample_config = {
                            'path_prefix': pf[0],
                            'anime_pred': pred[0],
                            'anime_true': yb[0]
                        }
                        sample_video(
                            sample_config,
                            os.path.join('sample_single', idx + '.mp4'))