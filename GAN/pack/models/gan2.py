from __future__ import absolute_import

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers
import matplotlib.pyplot as plt
from .models import UNet, ConvNet


default_config = {
    'b_size': 32,
    'img_h': 75,
    'img_w': 16,
    'z_dims': 16,
    'expr': 19,
    'penalty_scale': 10,
    'l1_scale': 3
}


def sample_z(s, m, n, l=-1., r=1.):
    return np.random.uniform(l, r, size=[s, m, n])


emo_noise = None


def generate_noise(noise_config):
    global emo_noise
    chr_noise = sample_z(1, 1, noise_config['chr_dims'])[0][0]
    if emo_noise is None:
        print('New emo noise.')
        emo_noise = sample_z(1, 1, noise_config['emo_dims'])[0][0]
    rdm_noises = sample_z(
        noise_config['length'],
        noise_config['time_steps'],
        noise_config['rdm_dims'],
        -0.05, 0.05)
    chr_noises = []
    emo_noises = []
    for _ in range(noise_config['length']):
        tmp_chr = []
        tmp_emo = []
        for _ in range(noise_config['time_steps']):
            tmp_chr.append(chr_noise)
            tmp_emo.append(emo_noise)
        chr_noises.append(np.asarray(tmp_chr))
        emo_noises.append(np.asarray(tmp_emo))
    chr_noises = np.asarray(chr_noises)
    emo_noises = np.asarray(emo_noises)
    # return np.concatenate((chr_noises, emo_noises, rdm_noises), axis=2)
    return chr_noises


class GAN():
    def __init__(self, scope='GAN', config=default_config):
        self._b_size = default_config['b_size']
        self._img_h = default_config['img_h']
        self._img_w = default_config['img_w']
        self._expr = default_config['expr']
        self._z_dims = default_config['z_dims']
        penalty_scale = default_config['penalty_scale']
        l1_scale = default_config['l1_scale']
        self.x = tf.placeholder(
            tf.float32, [self._b_size, self._img_h, self._img_w, 1])
        self.y = tf.placeholder(
            tf.float32, [self._b_size, self._img_h, self._expr, 1])
        self.z = tf.placeholder(
            tf.float32, [self._b_size, self._img_h, self._z_dims])
        Gy, Gy_logits = self.__G(self.x, self.z)
        Dr = self.__D(self.x, self.y)
        Df = self.__D(self.x, Gy, reuse=True)

        # penalty
        epsilon = tf.random_uniform([], 0.0, 1.0)
        self._penalty_y = epsilon * self.y + (1 - epsilon) * Gy
        Dp = self.__D(self.x, self._penalty_y, reuse=True)
        ddy = tf.gradients(Dp, self._penalty_y)[0]
        ddy = tf.sqrt(tf.reduce_sum(tf.square(ddy), axis=1))
        ddy = tf.reduce_mean(tf.square(ddy - 1.0) * penalty_scale)

        # loss for D
        WD = tf.reduce_mean(Dr) - tf.reduce_mean(Df) - ddy
        self.D_loss = -WD

        # loss for G
        L1 = tf.reduce_mean(
            tf.losses.absolute_difference(
                labels=self.y,
                predictions=Gy
            )
        )
        self.G_loss = -tf.reduce_mean(Df) + L1 * l1_scale

        self.pred = Gy
        self.L1_loss = L1
        # optim
        self.D_optim = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(
            self.D_loss,
            var_list=[
                var for var in tf.trainable_variables()
                if var.name.startswith("Dis")
            ]
        )
        self.G_optim = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(
            self.G_loss,
            var_list=[
                var for var in tf.trainable_variables()
                if var.name.startswith("Gen")
            ]
        )

        self.saver = tf.train.Saver()
        self.epoch_list = []
        self.loss_list = {
            'D_loss': [],
            'G_loss': [],
            'L1_train': [],
            'L1_valid': []
        }

    def __G(self, x, z, scope='Gen', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            net = UNet()
            # input 75x16, 3
            net.encode_layer(32, [1, 4], [1, 2], 'lrelu')
            # => 75x8, 32
            net.encode_layer(128, [8, 4], [5, 2], 'lrelu', True)
            # => 15x4, 128
            net.encode_layer(256, [5, 2], [3, 1], 'lrelu', True)
            # => 5x4, 256
            net.decode_layer(128, [5, 2], [3, 1], 'lrelu', True)
            # => 15x4, 128
            net.decode_layer(32, [8, 4], [5, 2], 'lrelu', True)
            # => 75x8, 32
            net.decode_layer(1, [1, 4], [1, 2], 'lrelu', True)
            # =>75x16, 1

            # concat noise at input
            z = tf.reshape(z, [self._b_size, self._img_h, self._img_w, -1])
            x = tf.concat([x, z], axis=3)
            logits = net(x)
            # concat noise at deocoded result
            # logits = tf.concat([logits, z], axis=3)
            logits = tflayers.fully_connected(
                tf.reshape(logits, [self._b_size*self._img_h, -1]),
                self._expr, None,
                weights_initializer=tf.random_normal_initializer(0, 0.02)
            )
            logits = tf.reshape(
                logits,
                [self._b_size, self._img_h, self._expr, 1]
            )
            anime = tf.sigmoid(logits)

            return anime, logits

    def __D(self, x, y, scope='Dis', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            paddings = [[0, 0], [0, 0], [0, 3], [0, 0]]
            xx = tf.pad(x, paddings)
            input = tf.concat([xx, y], axis=3)
            net = ConvNet()
            # input 75 x 19, 2
            net.conv_layer(32, [1, 4], [1, 2], 'lrelu')
            # => 75x10, 32
            net.conv_layer(64, [8, 4], [5, 2], 'lrelu', True)
            # => 15x5, 64
            net.conv_layer(128, [5, 2], [3, 1], 'lrelu', True)
            # => 5x5, 128
            lin = net(input)
            lin = tflayers.fully_connected(
                lin, 1, None,
                weights_initializer=tf.random_normal_initializer(0, 0.02)
            )
            return lin

    def sample(self, sess, path, id, vx, vy, vz, file_prefix):
        x_shape = [self._b_size, self._img_h, self._img_w, 1]
        y_shape = [self._b_size, self._img_h, self._expr, 1]
        z_shape = [self._b_size, self._img_h, self._z_dims]
        if sess is not None:
            if not os.path.exists(path):
                os.mkdir(path)
            pred = sess.run(self.pred, feed_dict={
                self.x: np.reshape(vx, x_shape),
                self.y: np.reshape(vy, y_shape),
                self.z: np.reshape(vz, z_shape)
            })
            from ..backend.utils.media import sample_video
            print('\nSampling...')
            sample_config = {
                'path_prefix': file_prefix[0],
                'anime_true': vy[0],
                'anime_pred': pred[0]
            }
            sample_video(
                sample_config,
                os.path.join(path, str(id) + '.mp4')
            )

    def train_batch(
            self, sess, epoch,
            x_batch, y_batch, z_batch,
            vx, vy, vz, file_prefix):
        x_shape = [self._b_size, self._img_h, self._img_w, 1]
        y_shape = [self._b_size, self._img_h, self._expr, 1]
        z_shape = [self._b_size, self._img_h, self._z_dims]
        x_batch = np.reshape(x_batch, x_shape)
        y_batch = np.reshape(y_batch, y_shape)
        z_batch = np.reshape(z_batch, z_shape)
        feed = {
            self.x: x_batch,
            self.y: y_batch,
            self.z: z_batch
        }
        valid_feed = {
            self.x: np.reshape(vx, x_shape),
            self.y: np.reshape(vy, y_shape),
            self.z: np.reshape(vz, z_shape)
        }
        if epoch % 500 == 0:
            if not os.path.exists('out'):
                os.mkdir('out')
            pred = sess.run(self.pred, feed_dict=valid_feed)
            from ..backend.utils.media import sample_video
            print('\nSampling...')
            sample_config = {
                'path_prefix': file_prefix[0],
                'anime_pred': pred[0]
            }
            if epoch == 0:
                sample_config['anime_true'] = vy[0]
            sample_video(
                sample_config,
                'out/epoch_' + str(epoch) + '.mp4'
            )
        D_loss = 0
        n_d = 100 if epoch < 25 or (epoch + 1) % 500 == 0 else 5
        for _ in range(n_d):
            loss, _ = sess.run(
                [self.D_loss, self.D_optim],
                feed_dict=feed
            )
            D_loss += loss
        D_loss /= n_d

        G_loss = 0
        for _ in range(1):
            loss, _ = sess.run(
                [self.G_loss, self.G_optim],
                feed_dict=feed
            )
            G_loss += loss
        G_loss /= 1

        L1_train = sess.run(
            self.L1_loss,
            feed_dict=feed
        )
        L1_valid = sess.run(
            self.L1_loss,
            feed_dict=valid_feed
        )

        info = 'Epoch ' + str(epoch) + ' '
        info += 'D_loss: ' + '%.4f' % D_loss + ' '
        info += 'G_loss: ' + '%.4f' % G_loss + ' '
        info += 'Train L1: ' + '%.4f' % L1_train + ' '
        info += 'Valid L1: ' + '%.4f' % L1_valid + '\r'
        sys.stdout.write(info)
        sys.stdout.flush()

        self.epoch_list.append(epoch)
        self.loss_list['D_loss'].append(D_loss)
        self.loss_list['G_loss'].append(G_loss)
        self.loss_list['L1_train'].append(L1_train)
        self.loss_list['L1_valid'].append(L1_valid)

        if epoch % 200 == 0:
            self.saver.save(sess, 'save/best.cpkt')

        if epoch % 5 == 0:
            fig = plt.figure(figsize=(12, 12))
            loss_plt = fig.add_subplot(211)
            loss_plt.title.set_text('Loss')
            l1_plt = fig.add_subplot(212)
            l1_plt.title.set_text('L1')
            loss_plt.plot(
                self.epoch_list, self.loss_list['D_loss'], 'g',
                self.epoch_list, self.loss_list['G_loss'], 'r'
            )
            l1_plt.plot(
                self.epoch_list, self.loss_list['L1_train'], 'g',
                self.epoch_list, self.loss_list['L1_valid'], 'r'
            )
            plt.savefig('error.png')
            plt.clf()
            plt.close(fig)


if __name__ == '__main__':
    gan = GAN()
