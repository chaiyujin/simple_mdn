import tensorflow as tf
from model import Model
import data_process

# data_process.process()

train_data = data_process.load('data/train.pkl')
valid_data = data_process.load('data/test.pkl')

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
        train_data=train_data,
        valid_data=valid_data,
        epoches=400,
        mini_batch_size=8,
        optimizer=optimizer
    )

