from __future__ import absolute_import

import sys
sys.path.append('..')
import tensorflow as tf
from utils import variable
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.distributions import Categorical
from tensorflow.contrib.distributions import Mixture
from tensorflow.contrib.distributions import MultivariateNormalDiag


def reshape_gmm_tensor(tensor, D, K):
    tmp = []
    for i in range(D):
        tmp.append(tensor[:, i * K: (i + 1) * K])
    return tf.stack(tmp, axis=2)


def parameter_layer(X, Dims, K):
    locs = fully_connected(X, K * Dims, activation_fn=None)
    scales = fully_connected(X, K * Dims, activation_fn=tf.exp)
    pi = fully_connected(X, K, activation_fn=tf.nn.softmax)
    # reshape the output tensor into parameters
    locs = reshape_gmm_tensor(locs, Dims, K)
    scales = reshape_gmm_tensor(scales, Dims, K)
    return locs, scales, pi


def mixture(locs, scales, pi, K):
    cat = Categorical(probs=pi)
    components = [
        MultivariateNormalDiag(loc=locs[:, i], scale_diag=scales[:, i])
        for i in range(K)]
    # get the mixture distribution
    mix = Mixture(cat=cat, components=components)
    return mix


def loss_fn(y, mixture):
    loss = tf.reduce_mean(-tf.log(mixture.prob(y)))
    return loss
