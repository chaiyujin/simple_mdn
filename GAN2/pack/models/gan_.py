from __future__ import absolute_import

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers
import matplotlib.pyplot as plt
from .models import ConvNet, UNet
from ..backend.train.RMSProp import RMSProp as MyRMS


default_config = {
    'b_size': 32,
    'time_steps': 75,
    'audio_dims': 16,
    'z_dims': 16,
    'expr_dims': 19,
    'penalty_scale': 10,
    'l1_scale': 3
}
start_noise_train_epoch = 15000


# Generator: input audio mfcc features and noise z
def generator(x, z, config, scope='Gen', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        if not reuse:
            print('Building generator.')
        else:
            print('Reuse generator.')
        x_s = [int(d) for d in x.get_shape()]
        z = tf.reshape(z, [x_s[0], x_s[1], x_s[2], -1])
        input = tf.concat([x, z], axis=3)
        if True:
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
            lin = tf.reshape(
                net(input),
                [x_s[0] * x_s[1], -1]
            )
            logits = tflayers.fully_connected(
                lin, config['expr_dims'], None,
                weights_initializer=tf.random_normal_initializer(0, 0.02)
            )
            logits = tf.reshape(
                logits, [x_s[0], x_s[1], config['expr_dims'], 1]
            )
            anime = tf.sigmoid(logits)

            return anime, logits


# Discriminator: input audio mfcc features and anime exprs
def discriminator(x, y, scope='Dis', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        # padding x, y to same shape
        x_dims = int(x.get_shape()[2])
        y_dims = int(y.get_shape()[2])
        input = None
        if not reuse:
            print('Building discriminator.')
            print('X dims:', x_dims, 'Y dims:', y_dims)
        else:
            print('Reuse discriminator.')
        if x_dims < y_dims:
            paddings = [[0, 0], [0, 0], [0, y_dims - x_dims], [0, 0]]
            xx = tf.pad(x, paddings)
            input = tf.concat([xx, y], axis=3)
        elif x_dims > y_dims:
            paddings = [[0, 0], [0, 0], [0, x_dims - y_dims], [0, 0]]
            yy = tf.pad(y, paddings)
            input = tf.concat([x, yy], axis=3)
        else:
            input = tf.concat([x, y], axis=3)

        # the network
        net = ConvNet() if reuse else ConvNet(config={'log'})
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


def get_penalty_data(x_true, x_fake):
    epsilon = tf.random_uniform([], 0.0, 1.0)
    penalty = epsilon * x_true + (1 - epsilon) * x_fake
    return penalty


def calc_penalty(D, penalty_x, penalty_scale):
    ddx = tf.gradients(D, penalty_x)[0]
    ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
    ddx = tf.reduce_mean(tf.square(ddx - 1.0) * penalty_scale)
    return ddx


class GAN():
    def __init__(self, scope='AudioGAN', config=default_config):
        self._b_size = config['b_size']
        self._img_h = config['time_steps']
        self._img_w = config['audio_dims']
        self._expr = config['expr_dims']
        self._z_dims = config['z_dims']
        penalty_scale = config['penalty_scale']
        l1_scale = config['l1_scale']
        # the shapes
        self.x_shape = [self._b_size, self._img_h, self._img_w, 1]
        self.y_shape = [self._b_size, self._img_h, self._expr, 1]
        self.z_shape = [self._z_dims]
        self.Z_shape = [self._b_size, self._img_h, self._z_dims]
        # the inputs
        self.x = tf.placeholder(tf.float32, self.x_shape)
        self.y = tf.placeholder(tf.float32, self.y_shape)
        self.z = tf.placeholder(tf.float32, self.z_shape)
        self.Z = tf.add(tf.zeros(self.Z_shape), self.z)
        # generator
        y_real = self.y
        y_fake, _ = generator(self.x, self.Z, config)
        self._p = get_penalty_data(y_real, y_fake)
        self.pred = y_fake
        # discriminator
        D_real = discriminator(self.x, y_real)
        D_fake = discriminator(self.x, y_fake, reuse=True)
        D_pena = discriminator(self.x, self._p, reuse=True)
        # penalty
        penalty = calc_penalty(D_pena, self._p, penalty_scale)

        # loss for D
        self.wasserstein =\
            tf.reduce_mean(D_real) - tf.reduce_mean(D_fake) - penalty
        self.D_loss = -self.wasserstein
        # loss for G
        L1 = tf.reduce_mean(tf.losses.absolute_difference(
            labels=y_real, predictions=y_fake
        ))
        self.G_loss = -tf.reduce_mean(D_fake) + l1_scale * L1
        self.L1_loss = L1

        # var_list
        self.D_theta = [
            var for var in tf.trainable_variables()
            if var.name.startswith("Dis")
        ]
        self.G_theta = [
            var for var in tf.trainable_variables()
            if var.name.startswith("Gen")
        ]

        self.grad_Dz = tf.gradients(self.D_loss, [self.z])[0]
        self.grad_Gz = tf.gradients(self.G_loss, [self.z])[0]
        self.grad_Dz /= self._img_h * self._b_size
        self.grad_Gz /= self._img_h * self._b_size

        # optim
        RMSProp = tf.train.RMSPropOptimizer(learning_rate=1e-4)
        self.D_optim = RMSProp.minimize(self.D_loss, var_list=self.D_theta)
        self.G_optim = RMSProp.minimize(self.G_loss, var_list=self.G_theta)
        self.z_optim = MyRMS(learning_rate=1e-6)

        # saver
        theta = [
            var for var in tf.trainable_variables()
            if var.name.startswith("Gen") or var.name.startswith("Dis")
        ]
        self.saver = tf.train.Saver(theta)

        # plot
        self.epoch_list = []
        self.loss_lists = {
            'D_loss': [], 'G_loss': [], 'L1_train': [], 'L1_valid': []
        }

    def _sample(self, sess, path, id, feed, prefix, ground=True):
        if not os.path.exists(path):
            os.mkdir(path)
        pred = sess.run(self.pred, feed_dict=feed)
        from ..backend.utils.media import sample_video
        print('\nSampling...')
        sample_config = {
            'path_prefix': prefix[0],
            'anime_pred': pred[0]
        }
        if ground:
            sample_config['anime_true'] = feed[self.y][0]
        sample_video(sample_config, os.path.join(path, str(id) + '.mp4'))

    def sample(self, sess, path, id, vx, vy, vz, file_prefix):
        feed_valid = self.generate_feed_dict(vx, vy, vz)
        self._sample(sess, path, id, feed_valid, file_prefix)

    def scale_sample(self, sess, path, id, vx, vy, vz, files):
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
            from ..backend.utils.media import sample_concat_video
            print('\nSampling...')
            sample_config = {
                'path_prefix': files,
                'anime_true': vy[0],
                'anime_pred': pred[0]
            }
            sample_concat_video(
                sample_config,
                os.path.join(path, str(id) + '.mp4')
            )

    def generate_feed_dict(self, x, y, z):
        x_batch = np.reshape(x, self.x_shape)
        y_batch = np.reshape(y, self.y_shape)
        z_batch = np.reshape(z, self.z_shape) if z is not None else None
        feed_d = {self.x: x_batch, self.y: y_batch}
        if z_batch is not None:
            feed_d[self.z] = z_batch
        return feed_d

    def train_batch(
            self, sess, epoch, id, x, y, z,
            vx, vy, vz, prefix, only_train_noise=False):
        feed_valid = self.generate_feed_dict(vx, vy, vz)
        # sample
        if epoch % 500 == 0:
            self._sample(
                sess, 'out', str(epoch),
                feed_valid, prefix, (epoch == 0)
            )

        new_z = z
        # BP
        D_loss = 0
        n_d = 100 if epoch < 25 or (epoch + 1) % 500 == 0 else 5
        for _ in range(n_d):
            feed_train = self.generate_feed_dict(x, y, new_z)
            if only_train_noise:
                dDz, loss = sess.run(
                    [self.grad_Dz, self.D_loss], feed_dict=feed_train
                )
            else:
                dDz, loss, _ = sess.run(
                    [self.grad_Dz, self.D_loss, self.D_optim],
                    feed_dict=feed_train
                )
            D_loss += loss
            if only_train_noise or epoch > start_noise_train_epoch:
                new_z = self.z_optim.apply_gradient(new_z, dDz, 'Dz' + str(id))
        D_loss /= n_d

        G_loss = 0
        for _ in range(1):
            feed_train = self.generate_feed_dict(x, y, new_z)
            if only_train_noise:
                dGz, loss = sess.run(
                    [self.grad_Gz, self.G_loss], feed_dict=feed_train
                )
            else:
                dGz, loss, _ = sess.run(
                    [self.grad_Gz, self.G_loss, self.G_optim],
                    feed_dict=feed_train
                )
            G_loss += loss
            if only_train_noise or epoch > start_noise_train_epoch:
                new_z = self.z_optim.apply_gradient(new_z, dGz, 'Gz' + str(id))
        G_loss /= 1

        if epoch % 100 == 0:
            if (not only_train_noise):
                if not os.path.exists('save'):
                    os.mkdir('save')
                self.saver.save(sess, 'save/best.cpkt')

        if epoch % 50 == 0:
            L1_train = sess.run(self.L1_loss, feed_dict=feed_train)
            L1_valid = sess.run(self.L1_loss, feed_dict=feed_valid)

            info = 'Epoch ' + str(epoch) + ' '
            info += 'D_loss: ' + '%.4f' % D_loss + ' '
            info += 'G_loss: ' + '%.4f' % G_loss + ' '
            info += 'Train L1: ' + '%.4f' % L1_train + ' '
            info += 'Valid L1: ' + '%.4f' % L1_valid + '\r'
            sys.stdout.write(info)
            sys.stdout.flush()

            self.epoch_list.append(epoch)
            self.loss_lists['D_loss'].append(D_loss)
            self.loss_lists['G_loss'].append(G_loss)
            self.loss_lists['L1_train'].append(L1_train)
            self.loss_lists['L1_valid'].append(L1_valid)

            fig = plt.figure(figsize=(12, 12))
            loss_plt = fig.add_subplot(211)
            loss_plt.title.set_text('Loss')
            l1_plt = fig.add_subplot(212)
            l1_plt.title.set_text('L1')
            loss_plt.plot(
                self.epoch_list, self.loss_lists['D_loss'], 'g',
                self.epoch_list, self.loss_lists['G_loss'], 'r'
            )
            l1_plt.plot(
                self.epoch_list, self.loss_lists['L1_train'], 'g',
                self.epoch_list, self.loss_lists['L1_valid'], 'r'
            )
            plt.savefig('error.png')
            plt.clf()
            plt.close(fig)

        return new_z
