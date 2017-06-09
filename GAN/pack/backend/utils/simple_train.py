from __future__ import absolute_import

import sys
from . import process_bar


def train_model(
        sess, optimizer, loss,
        X_tensor, Y_tensor,
        X_train, Y_train,
        X_valid, Y_valid,
        iterations, batch_size=-1):
    num_examples = len(X_train)
    if batch_size < 0:
        batch_size = num_examples
    num_batch = int((num_examples - 1) / batch_size) + 1
    for iter in range(iterations):
        for batch in range(num_batch):
            indexes = [i % num_examples
                       for i in range(batch * batch_size,
                                      (batch + 1) * batch_size)]

            bar = process_bar.process_bar(iter, iterations)
            # train batch
            feed_train = {X_tensor: X_train[indexes],
                          Y_tensor: Y_train[indexes]}
            train_loss, _ = sess.run([loss, optimizer], feed_dict=feed_train)

            # valid
            feed_valid = {X_tensor: X_valid,
                          Y_tensor: Y_valid}
            valid_loss = sess.run(loss, feed_dict=feed_valid)

            train_loss_str = "%.4f" % train_loss
            valid_loss_str = "%.4f" % valid_loss

            bar += ' Train Loss:' + train_loss_str +\
                ' Valid Loss:' + valid_loss_str + '\r'
            sys.stdout.write(bar)
            sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
