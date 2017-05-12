import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

# the function try to learn
N = 200
X = np.linspace(0, 1, N)
Y = X + 0.3 * np.sin(2 * math.pi * X) + np.random.uniform(-0.1, 0.1, N)
X, Y = Y, X
# draw it
# plt.scatter(X, Y, color='g')
# plt.show()

# parameters
input_size = 1
output_size = 1
hidden_size = 30
M = 3  # number of mixure models
batch_size = 200


def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.25)
    return tf.Variable(inital)


def bias_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(inital)


x = tf.placeholder('float32', shape=[None, input_size])
y = tf.placeholder('float32', shape=[None, output_size])

W = {}
b = {}
W['xh'] = weight_variable([input_size, hidden_size])
W['hu'] = weight_variable([hidden_size, M])
W['hs'] = weight_variable([hidden_size, M])
W['hp'] = weight_variable([hidden_size, M])

b['xh'] = bias_variable([1, hidden_size])
b['hu'] = bias_variable([1, M])
b['hs'] = bias_variable([1, M])
b['hp'] = bias_variable([1, M])

# the graph
hidden_layer = tf.nn.tanh(tf.matmul(x, W['xh']) + b['xh'])
mu = tf.matmul(hidden_layer, W['hu']) + b['hu']
sigma = tf.exp(tf.matmul(hidden_layer, W['hs']) + b['hs'])
pi = tf.nn.softmax(tf.matmul(hidden_layer, W['hp']) + b['hp'])  # [nb, M]
sum_pi = tf.reduce_sum(pi, 1, keep_dims=True)
# NLL
# single gm
ds = tf.contrib.distributions
cat = ds.Categorical(logits=pi)
components = [
    ds.Normal(loc=loc, scale=scale) for loc, scale
              in zip(tf.unstack(tf.transpose(mu)),
                     tf.unstack(tf.transpose(sigma)))]

print(cat.batch_shape)
print(components[0].batch_shape)

mix = ds.Mixture(cat=cat, components=components)
pg = mix.prob(y)
# gauss = tf.exp(-(tf.pow(y - mu, 2)) / (2 * tf.pow(sigma, 2))) / \
#         (sigma * np.sqrt(2 * math.pi))  # [nb, M]
# gm * possibility
# pg = gauss * pi  # [nb, M]
# negative log of pdf
lp = -tf.log(tf.reduce_sum(pg, 1, keep_dims=True))  # [nb, 1]
# sum up log of pdf
loss = tf.reduce_mean(lp)


# train
nb = N  # full batch
xbatch = np.reshape(X[:nb], (nb, 1))
ybatch = np.reshape(Y[:nb], (nb, 1))
train_step = tf.train.AdagradOptimizer(1e-2).minimize(loss)
sess.run(tf.global_variables_initializer())

print('Begin to train')
for i in range(20000):
    if i % 1000 == 0:
        train_loss = loss.eval(feed_dict={x: xbatch, y: ybatch})
        print("step %d, training loss %g" % (i, train_loss))
        # print(sum_pi.eval(feed_dict={x: xbatch}))
    train_step.run(feed_dict={x: xbatch, y: ybatch})

print("training loss %g" % loss.eval(feed_dict={x: xbatch, y: ybatch}))


def sample(mus, sigmas, pis):
    best = pis.argmax(axis=1)  # [nb x 1]
    print(best.shape)

    # select the best
    indices = np.zeros_like(mus)  # [nb x M]
    indices[range(mus.shape[0]), best] = 1

    best_mus = np.sum(np.multiply(indices, mus), axis=1)
    best_sigmas = np.sum(np.multiply(indices, sigmas), axis=1)

    Y_ = np.random.normal(best_mus, best_sigmas)
    return Y_


# plot results
X_ = xbatch
mus = mu.eval(feed_dict={x: xbatch})  # [nb x M]
sigmas = sigma.eval(feed_dict={x: xbatch})  # [nb x M]
pis = pi.eval(feed_dict={x: xbatch})  # [nb x M]
Y_ = sample(mus, sigmas, pis)
print(X_)
plt.scatter(X, Y_, color='g')

plt.show()
