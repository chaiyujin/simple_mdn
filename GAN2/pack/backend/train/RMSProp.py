'''
    learning_rate,
    decay=0.9,
    momentum=0.0,
    epsilon=1e-10,
'''
import math


class RMSProp():
    def __init__(self, learning_rate, decay=0.9, epsilon=1e-10):
        self._lr = learning_rate
        self._decay = decay
        self._eps = epsilon
        self._mean_squares = {}

    def apply_gradient(self, var, gradient, var_name):
        # get from _mean_squares
        r = 0
        if var_name in self._mean_squares:
            r = self._mean_squares[var_name]
        # calc
        r = self._decay * r + (1 - self._decay) * (gradient * gradient)
        delta = -self._lr * gradient / math.sqrt(self._eps + r)
        var = var + delta

        # save into var map
        self._mean_squares[var_name] = r

        return var


if __name__ == '__main__':
    def f(x):
        return x * x + 2 * x + 1

    def grad(x):
        return 2 * x + 2

    rms = RMSProp(1e-4)

    x = 10
    epoch = 0
    while True:
        y = f(x)
        x = rms.apply_gradient(x, grad(x), 'x')
        if epoch % 1000 == 0:
            print(x, ':', y)
            print(rms._mean_squares)
        epoch += 1
