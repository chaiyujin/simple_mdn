from __future__ import absolute_import

import os
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from ..utils import media, console


def find_files(path, target_ext):
    if target_ext[0] != '.':
        target_ext = '.' + target_ext
    result_list = []
    for parent, dirs, files in os.walk(path):
        for file in files:
            name, ext = os.path.splitext(os.path.join(parent, file))
            if ext == target_ext:
                result_list.append(name + ext)
    return result_list


def generate_train_test(path, ext, rate=0.2, config=None):
    file_list = find_files(path, ext)
    train_list = []
    test_list = []
    for file_path in file_list:
        if random.random() < rate:
            test_list.append(file_path)
        else:
            train_list.append(file_path)
    print('Train data: ' + str(len(train_list)))
    print('Test data: ' + str(len(test_list)))

    train_set = accumulate_data(train_list, config)
    test_set = accumulate_data(test_list, config)
    return {
        'train_set': train_set,
        'test_set': test_set
    }


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % 
                             truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is\
                              different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths


def accumulate_data(file_list, config):
    inputs = []
    outputs = []
    seq_len = []
    silence = []
    path_prefix = []
    if len(file_list) == 0:
        return None
    failed = 0
    for idx, file in enumerate(file_list):
        console.log(
            'info', 'Process ' + str(idx + 1) + '/' + str(len(file_list)))
        res = None
        try:
            res = media.process_media(file, config)
        except:
            res = None
        if res is not None:
            assert(len(res[0]) == 75)
            inputs.append(res[0])
            outputs.append(res[1])
            seq_len.append(len(res[0]))
            silence.append(res[2])
            path_prefix.append(res[3])
        else:
            failed += 1
        # if idx > 10: break
    if failed > 0:
        console.log('error', 'Failed', str(failed) + '\n')
    if len(inputs) == 0:
        return None
    # padding the data at end
    # inputs, _ = pad_sequences(inputs, dtype=np.float64)
    # outputs, seq_len = pad_sequences(outputs, dtype=np.float32)
    inputs = np.asarray(inputs, dtype=np.float32)
    outputs = np.asarray(outputs, dtype=np.float32)
    seq_len = np.asarray(seq_len, dtype=np.int32)
    silence = np.asarray(silence, dtype=np.float32)
    assert(inputs.dtype == np.float32)
    assert(outputs.dtype == np.float32)

    return {
        'inputs': inputs,
        'outputs': outputs,
        'seq_len': seq_len,
        'silence': silence,
        'path_prefix': path_prefix
    }


def save(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def process(
        config,
        root_path='../../dataset/GRID/video/s1/', ext='mpg', test_rate=0.2,
        train_path='data/train.pkl', test_path='data/test.pkl'):

    if os.path.exists(train_path) and os.path.exists(test_path):
        res = ''
        while res.lower() != 'y' and res.lower() != 'n':
            res = input('Find train and test sets, replace? [y/[N]]').lower()
            if res == '':
                res = 'n'
        if res == 'n':
            return

    media.init_dde_fbx()
    sets = generate_train_test(root_path, ext, test_rate, config)
    save(train_path, sets['train_set'])
    save(test_path, sets['test_set'])


def merge(path_list, path):
    all_set = {}
    for p in path_list:
        with open(p, 'rb') as file:
            the_set = pickle.load(file)
            for k in the_set:
                if k in all_set:
                    all_set[k] = np.append(all_set[k], the_set[k], axis=0)
                else:
                    all_set[k] = np.asarray(the_set[k])
    with open(path, 'wb') as file:
        pickle.dump(all_set, file)


if __name__ == '__main__':
    # process()
    train_data = load('data/train.pkl')
    data = train_data['outputs']
    train_data['label'] = []
    count = 0
    all = 0
    for i in range(data.shape[0]):
        # d = data[i] ** 0.5
        d = data[i]
        label = []
        for j in range(d.shape[0]):
            max_v = np.amax(d[j])
            if max_v < 0.04:
                print('!')
                count += 1
            all += 1
    print(count, '/', all)
    # data = data ** 0.25
    print(data.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(data, bins=20)
    plt.show()
