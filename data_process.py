from __future__ import absolute_import

import os
import sys
import dde
import random
import fbxanime
import numpy as np
import pickle
from utils import media


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


def generate_train_test(path, ext, rate=0.2):
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

    train_set = accumulate_data(train_list)
    test_set = accumulate_data(test_list)
    return {
        'train_set': train_set,
        'test_set': test_set
    }


def accumulate_data(file_list):
    inputs = []
    outputs = []
    seq_len = []
    if len(file_list) == 0:
        return None
    failed = 0
    for idx, file in enumerate(file_list):
        sys.stdout.write('\033[01;33m[Process ' + str(idx + 1) + '/'
                         + str(len(file_list)) + ']\033[0m ')
        sys.stdout.flush()
        res = media.process_media(file)
        if res is not None:
            inputs.append(res[0])
            outputs.append(res[1])
            seq_len.append(len(res[0]))
        else:
            failed += 1
        if idx > 2: break
    if failed > 0:
        print('\033[01;31m[Failed]\033[0m ' + str(failed))
    if len(inputs) == 0:
        return None
    inputs = np.asarray(inputs, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float32)
    seq_len = np.asarray(seq_len, dtype=np.int32)
    assert(inputs.dtype == np.float64)
    assert(outputs.dtype == np.float32)

    return {
        'inputs': inputs,
        'outputs': outputs,
        'seq_len': seq_len
    }


def save(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def init(
    dde_path='D:/software/dev/DDE/v3.bin',
    fbx_path='asset/fbx_anime.fbx'
):
    dde.init(dde_path)
    fbxanime.init(640, 480, fbx_path)


def process(
        root_path='../../dataset/GRID/video/s1/', ext='mpg', test_rate=0.2,
        train_path='data/train.pkl', test_path='data/test.pkl'):

    init()
    sets = generate_train_test(root_path, ext, test_rate)
    save(train_path, sets['train_set'])
    save(test_path, sets['test_set'])


if __name__ == '__main__':
    process()
