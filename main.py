import tensorflow as tf
from model import Model
from data_process import load

data = load('data/train.pkl')
# print(len(data['inputs']))

if __name__ == '__main__':
    config = {
        'audio_num_features': 13,
        'anime_num_features': 19,
        'audio_lstm_size': 128,
        'anime_lstm_size': 100,
        'dense_size': 500,
        'mdn_K': 20,
        'phoneme_classes': 40,
        'train': 1,
        'dropout': 0.5
    }
    model = Model(config)

    optimizer = tf.train.MomentumOptimizer(1e-4, 0.9)
    model.simple_train(
        audio=data['inputs'],
        anime=data['outputs'],
        seq_len=data['seq_len'],
        epoches=200,
        mini_batch_size=8,
        optimizer=optimizer
    )

