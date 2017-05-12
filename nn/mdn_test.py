import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import slim
from sklearn.model_selection import train_test_split
from tensorflow.contrib.distributions import Categorical, Normal, Mixture, MultivariateNormalDiag

sess = tf.InteractiveSession()

N = 5000  # number of data points
D = 2  # number of features
K = 20  # number of mixture components


def build_toy_dataset(N):
    y_data = np.random.uniform(-10.5, 10.5, N * D)
    y_data = y_data.reshape((N, D))
    r_data = np.random.normal(size=N)  # random noise
    x_data = np.sin(0.75 * y_data[:, 0]) * 7.0 + y_data[:, 0] * 0.5 + r_data * 1.0
    x_data = x_data.reshape((N, 1))
    return train_test_split(x_data, y_data, random_state=42)


X_train, X_test, y_train, y_test = build_toy_dataset(N)
print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))
sns.regplot(X_train, y_train[:, 0], fit_reg=False)
plt.show()

X_ph = tf.placeholder(tf.float32, [None, 1])
y_ph = tf.placeholder(tf.float32, [None, D])


def reshape_gmm_tensor(tensor, D, K):
    tmp = []
    for i in range(D):
        tmp.append(tensor[:, i * K: (i + 1) * K])
    return tf.stack(tmp, axis=2)


def neural_network(X):
    """loc, scale, logits = NN(x; theta)"""
    # 2 hidden layers with 15 hidden units
    hidden1 = slim.fully_connected(X, 15)
    hidden2 = slim.fully_connected(hidden1, 15)
    locs = slim.fully_connected(hidden2, K * D, activation_fn=None)
    scales = slim.fully_connected(hidden2, K * D, activation_fn=tf.exp)
    logits = slim.fully_connected(hidden2, K, activation_fn=None)
    return reshape_gmm_tensor(locs, D, K), reshape_gmm_tensor(scales, D, K), logits


locs, scales, logits = neural_network(X_ph)
cat = Categorical(logits=logits)
components = [MultivariateNormalDiag(loc=locs[:, i], scale_diag=scales[:, i]) for i in range(K)]
mix = Mixture(cat=cat, components=components)
print(cat.batch_shape)
print(mix.batch_shape)
pg = mix.prob(y_ph)
print(pg.shape)
lp = -tf.log(pg)
loss = tf.reduce_mean(lp)

nb = N  # full batch
xbatch = X_train
ybatch = y_train
train_step = tf.train.AdagradOptimizer(1e-2).minimize(loss)
sess.run(tf.global_variables_initializer())

print('Begin to train')
for i in range(20000):
    # print(locs[:, :, tf.newaxis][:, 0].eval(feed_dict={X_ph: xbatch}).shape)
    # print(scales[:, :, tf.newaxis][:, 0].eval(feed_dict={X_ph: xbatch}).shape)
    # print(X_ph.eval(feed_dict={X_ph: xbatch}).shape)
    # print(y_ph.eval(feed_dict={y_ph: ybatch}).shape)
    # print(components[0].prob(y_ph).eval(feed_dict={X_ph: xbatch, y_ph: ybatch}).shape)
    # print(pg.eval(feed_dict={X_ph: xbatch, y_ph: ybatch}).shape)
    if i % 1000 == 0:
        train_loss = loss.eval(feed_dict={X_ph: xbatch, y_ph: ybatch})
        print("step %d, training loss %g" % (i, train_loss))
        # print(sum_pi.eval(feed_dict={x: xbatch}))
    train_step.run(feed_dict={X_ph: xbatch, y_ph: ybatch})

print("training loss %g" % loss.eval(feed_dict={X_ph: xbatch, y_ph: ybatch}))


def sample_from_mixture(x, pred_weights, pred_means, pred_std, amount):
    """Draws samples from mixture model.

    Returns 2 d array with input X and sample from prediction of mixture model.
    """
    samples = np.zeros((amount, 2))
    n_mix = len(pred_weights[0])
    to_choose_from = np.arange(n_mix)
    for j, (weights, means, std_devs) in enumerate(
            zip(pred_weights, pred_means, pred_std)):
        index = np.random.choice(to_choose_from, p=weights)
        samples[j, 1] = np.random.normal(means[index, 0], std_devs[index, 0], size=1)
        samples[j, 0] = x[j]
        if j == amount - 1:
            break
    return samples


pred_weights, pred_means, pred_std = \
    sess.run([tf.nn.softmax(logits), locs, scales], feed_dict={X_ph: X_test})

a = sample_from_mixture(X_test, pred_weights, pred_means, pred_std, amount=len(X_test))
sns.jointplot(a[:, 0], a[:, 1], color="#4CB391", ylim=(-10, 10), xlim=(-14, 14))
plt.show()
