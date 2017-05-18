import numpy
import data_process
import tensorflow as tf
from model import Model
from utils.media import sample_video
from utils.process_bar import process_bar
from utils import console


def sample(sess, model, data, idx):
    audio = data['inputs'][idx: idx + 1]
    anime_true = data['outputs'][idx]
    assert(len(audio[0]) == len(anime_true))
    path_prefix = data['path_prefix'][idx]

    anime_pred = model.sample_audio(sess, audio)
    anime_pred, _ = data_process.pad_sequences(
        anime_pred, 19
    )
    anime_true, _ = data_process.pad_sequences(
        anime_true, 19
    )
    # print(anime_pred)
    # print(anime_true)
    assert(len(anime_pred) == len(anime_true))
    sample_video(
        {
            'path_prefix': path_prefix,
            'anime_pred': anime_pred,
            'anime_true': anime_true
        },
        'result/' + str(idx) + '.mp4')


if __name__ == '__main__':
    config = {
        'audio_num_features': 39,
        'anime_num_features': 1,
        'audio_lstm_size': 50,
        'anime_lstm_size': 50,
        'dense_size': 100,
        'mdn_K': 3,
        'phoneme_classes': 20,
        'train': 1,
        'dropout': 0.5
    }

    model = Model(config)

    data_process.process(config)
    train_data = data_process.load('data/train.pkl')
    valid_data = data_process.load('data/test.pkl')
    mean = numpy.mean(train_data['inputs'])
    stdv = numpy.std(train_data['inputs'])
    train_data['inputs'] = (train_data['inputs'] - mean) / stdv
    valid_data['inputs'] = (valid_data['inputs'] - mean) / stdv
    print(mean, stdv)

    if model._train:
        optimizer = tf.train.MomentumOptimizer(1e-3, 0.9)
        model.simple_train(
            train_data=train_data,
            valid_data=valid_data,
            epoches=500,
            mini_batch_size=4,
            optimizer=optimizer
        )

    with tf.Session() as sess:
        model.load(sess)
        # model.run_one_epoch(
        #     sess, valid_data, 8, None
        # )
        for i in range(10):
            bar = process_bar(i, 10)
            console.log('', '', bar + '\r')
            sample(sess, model, valid_data, i)
        console.log()
