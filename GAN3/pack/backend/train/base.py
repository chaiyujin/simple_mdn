from __future__ import absolute_import


class BasicTrainer():
    def __init__(self, model, train_set, valid_set, label_key, feed_keys=None):
        self._model = model
        self._train_set = train_set
        self._valid_set = valid_set
        self._key = feed_keys
        self._label_key = label_key
        self._dict = self._model.placeholder_dict
        if self._key is None:
            self._key = {}
            for k in self._dict:
                self._key[k] = k
        else:
            for k in self._dict:
                if not (k in self._key):
                    raise ValueError('feed_keys do not give "' + k + '"')

    def train(self, sess, optimizer, epoches,
              mini_batch_size, valid_batch_size,
              load=False):
        raise NotImplementedError()

    @property
    def train_set(self):
        return self._train_set

    @property
    def valid_set(self):
        return self._valid_set

    @property
    def label_key(self):
        return self._label_key

    def next_feed_dict(self, set):
        sub_set, _ = set.next_batch()
        if sub_set is None:
            return None, None
        feed = {}
        for k in self._dict:
            feed[self._dict[k]] = sub_set[self._key[k]]
        return feed, sub_set
