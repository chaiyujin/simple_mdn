from __future__ import absolute_import

import os
import math
import warnings
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle


class DataSet():
    def __init__(self, keys=None):
        self._batch_size = 1    # default batch_size is 1
        self._keys = keys
        self._normalized = {}
        self._powered = []
        self.clear()

    def add_pkl(self, path):
        if not (os.path.exists(path)):
            return False
        with open(path, 'rb') as pkl:
            data = pickle.load(pkl)
            count = 0
            if self._keys is None:
                count = len(data)
            else:
                for k in self._keys:
                    if not (k in data):
                        raise ValueError('pkl does not has key ' + k)
                count = len(data[self._keys[0]])
            # update the records
            self._pkls.append({
                'path': path,
                'count': count
            })
            self._total_tuples += count
        return True

    # clear the dataset
    def clear(self):
        self._pkls = []        # the paths to load pickle
        self._total_tuples = 0
        self.reset()

    # reset the position of batch
    def reset(self):
        self._pkl_idx = 0      # the current idx point to paths
        self._pkl_curr = 0      # the current position in the current file
        self._batch_idx = 0     # the batch idx
        self._cache = None      # cache the pickle

    # return (next_batch, batch_size)
    def next_batch(self):
        lc = self._batch_size * self._batch_idx
        rc = lc + self._batch_size
        if rc > self._total_tuples:
            rc = self._total_tuples
        need = rc - lc
        if need <= 0:
            return None, 0
        # begin to read
        if self._cache is None:
            # get cache pkl
            self.__load_cache()
        ret = None
        # check if the cached is enough
        if self._pkl_curr + need <= self._pkls[self._pkl_idx]['count']:
            ret = self.__load_from_cache(self._pkl_curr, need)
            # update position
            self._batch_idx += 1
            self._pkl_curr += need
        else:
            give = self._pkls[self._pkl_idx]['count'] - self._pkl_curr
            ret = self.__load_from_cache(self._pkl_curr, give)
            # read next pkl
            self._pkl_idx += 1
            self.__load_cache()
            tmp = self.__load_from_cache(0, need - give)
            ret = self.__extend_result(ret, tmp)
            # update position
            self._pkl_curr = need - give
            self._batch_idx += 1
        # normalize
        for key in self._normalized:
            norm = self._normalized[key]
            ret[key] = (ret[key] - norm['mean']) / norm['stdv']
        # power
        for power in self._powered:
            ret[power['key']] = ret[power['key']] ** power['value']
        return ret, need

    def power(self, key, p):
        self._powered.append({
            'key': key,
            'value': p
        })

    # normalize the certain data
    def normalize(self, key, mean=None, stdv=None):
        assert(self._keys is not None)
        assert(key in self._keys)
        # already normalized
        if key in self._normalized:
            warnings.warn('The key <' + key + '> is already normalized.')
            return

        if mean is None:
            self.reset()
            mean = 0
            cnt = 0
            while True:
                batch, s = self.next_batch()
                if batch is None:
                    break
                mean += batch[key].mean() * s
                cnt += s
            mean /= cnt

        if stdv is None:
            self.reset()
            stdv = 0
            cnt = 0
            while True:
                batch, s = self.next_batch()
                if batch is None:
                    break
                stdv += ((batch[key] - mean) ** 2).mean() * s
                cnt += s
            stdv /= cnt
            stdv = math.sqrt(stdv)
        self.reset()
        self._normalized[key] = {
            'mean': mean,
            'stdv': stdv
        }
        return mean, stdv

    def entire_set(self):
        ret = None
        self.reset()
        while True:
            batch, _ = self.next_batch()
            if batch is None:
                break
            if ret is None:
                ret = batch
            else:
                ret = self.__extend_result(ret, batch)
        self.reset()
        return ret

    def __extend_result(self, a, b):
        if a is None:
            return b
        if b is None:
            return a
        if self._keys is None:
            return np.append(a, b, axis=0)
        else:
            ret = {}
            for k in self._keys:
                ret[k] = np.append(a[k], b[k], axis=0)
            return ret

    def __load_cache(self):
        path = self._pkls[self._pkl_idx]['path']
        with open(path, 'rb') as pkl:
            self._cache = pickle.load(pkl)

    def __load_from_cache(self, start, length):
        if self._cache is None:
            return None
        if self._keys is None:
            return self._cache[start: start + length]
        else:
            ret = {}
            for k in self._keys:
                ret[k] = self._cache[k][start: start + length]
            return ret

    def __get_batch_size(self):
        return self._batch_size

    def __set_batch_size(self, value):
        size = int(value)
        if size > self._total_tuples:
            size = self._total_tuples
        if size < 1:
            size = 1
        self._batch_size = size

    batch_size = property(__get_batch_size, __set_batch_size)

    @property
    def length(self):
        return self._total_tuples


class DataLoader():
    def __init__(self):
        pass


if __name__ == '__main__':
    import sys
    set = DataSet(keys=['inputs', 'outputs', 'seq_len'])
    set.add_pickle('data/train.pkl')
    set.add_pickle('data/test.pkl')
    set.batch_size = 100
    set.reset()
    count = 0
    valid = None
    while True:
        batch, size = set.next_batch()
        if batch is None:
            break
        valid = batch
        count += size
        sys.stdout.write("%d\r" % count)
        sys.stdout.flush()
    print()
    print(len(valid['inputs']))

    set.reset()
    count = 0
    valid = None
    while True:
        batch, size = set.next_batch()
        if batch is None:
            break
        valid = batch
        count += size
        sys.stdout.write("%d\r" % count)
        sys.stdout.flush()
    print()
    print(len(valid['inputs']))
