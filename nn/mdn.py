from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.distributions import Categorical
from tensorflow.contrib.distributions import Mixture
from tensorflow.contrib.distributions import MultivariateNormalDiag


def reshape_gmm_tensor(tensor, D, K):
    tmp = []
    for i in range(D):
        tmp.append(tensor[:, :, i * K: (i + 1) * K])
    stacked = tf.stack(tmp, axis=3)
    reshaped = tf.reshape(
        stacked,
        [-1, K, D])
    return reshaped


def parameter_layer(X, Dims, K, bias=0):
    locs = fully_connected(X, K * Dims, activation_fn=tf.sigmoid)
    scales_hat = fully_connected(X, K * Dims, activation_fn=None)
    pi_hat = fully_connected(X, K, activation_fn=None)
    # add bias on the parameter
    # larger bias, more stable
    if bias > 0:
        b_scale = tf.constant(
            np.full((K * Dims), -bias, np.float32),
            dtype=tf.float32
        )
        b_pi = tf.constant(
            np.full((K), 1 + bias, np.float32),
            dtype=tf.float32
        )
        scales = tf.sigmoid(tf.add(scales_hat, b_scale))
        pi = tf.nn.softmax(tf.multiply(pi_hat, b_pi))
    else:
        scales = tf.sigmoid(scales_hat)
        pi = tf.nn.softmax(pi_hat)
    # scale should be bigger than 0.05
    # otherwise, the loss will boom
    # min_scale = tf.constant(
    #     np.full((K * Dims), 0.05, np.float32),
    #     dtype=tf.float32
    # )
    # scales = tf.add(scales, min_scale)
    scales = tf.clip_by_value(scales, 0.05, 0.4)
    # reshape the output tensor into parameters
    locs = reshape_gmm_tensor(locs, Dims, K)
    scales = reshape_gmm_tensor(scales, Dims, K)
    pi = tf.reshape(pi, [-1, K])
    scales_hat = reshape_gmm_tensor(scales_hat, Dims, K)
    pi_hat = tf.reshape(pi_hat, [-1, K])
    return locs, scales, pi, scales_hat, pi_hat


def mixture(locs, scales, pi, K):
    cat = Categorical(probs=pi)
    components = [
        MultivariateNormalDiag(loc=locs[:, i, :], scale_diag=scales[:, i, :])
        for i in range(K)]
    # get the mixture distribution
    mix = Mixture(cat=cat, components=components)
    return mix


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sample directly from locs, scale_diag and pi
def sample_gmm(locs, scales, pi):
    idx = np.random.choice(len(pi), p=pi)
    loc = locs[idx, :]
    scale = scales[idx, :]
    scale_diag = np.zeros((scale.shape[0], scale.shape[0]))
    for i in range(len(scale)):
        scale_diag[i][i] = scale[i]
    x = np.random.multivariate_normal(loc, scale_diag, 1)
    return x[0]


# use hat result to sample, with bias
def sample_gmm_with_hat(locs, scales_hat, pi_hat, bias=0):
    b_scale = np.full(scales_hat.shape, -bias, np.float32)
    b_pi = np.full(pi_hat.shape, 1 + bias, np.float32)
    scales = sigmoid(scales_hat + b_scale)
    pi = softmax(pi_hat * b_pi)
    return sample_gmm(locs, scales, pi)


def loss_fn(y, mixture):
    # prob = mixture.prob(y)
    loss = tf.reduce_mean(
        # -tf.log(
        #     prob
        # )
        -mixture.log_prob(y)
    )
    return loss
