from __future__ import absolute_import

import os
import sys
import dde
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


def accumulate_data(path, ext):
    file_list = find_files(path, ext)
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
        # if idx > 20: break
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


if __name__ == '__main__':
    init()
    d = accumulate_data('../../dataset/GRID/video/s1/', 'mpg')
    # d = accumulate_data('./', 'mpg')
    save('data/train.pkl', d)
