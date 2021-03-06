from __future__ import absolute_import

import tensorflow as tf


def LSTM_cell(size, init, dropout=0):
    if init is None:
        init = tf.orthogonal_initializer
    cell = tf.contrib.rnn.LSTMCell(
        size, state_is_tuple=True,
        initializer=init
    )
    if dropout > 0:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=(1.0 - dropout)
        )
    return cell


def LSTM_layer(
        batch_size, lstm_size, inputs, seq_len,
        initializer=None, dropout=0, scope='lstm_layer'):
    layer = {}
    layer['cell'] = LSTM_cell(lstm_size, initializer, dropout)
    layer['init_state'] = layer['cell'].zero_state(
        batch_size=batch_size, dtype=tf.float32
    )
    layer['output'], layer['last_state'] = tf.nn.dynamic_rnn(
        cell=layer['cell'], inputs=inputs, sequence_length=seq_len,
        scope=scope, dtype=tf.float32, initial_state=layer['init_state']
    )
    return layer


def dense_layer(
        input_size, output_size, inputs,
        initializer=None, activation=None, scope='dense_layer'):
    layer = {}
    w_init = initializer
    b_init = initializer
    if w_init is None:
        w_init = tf.contrib.layers.xavier_initializer()
    if b_init is None:
        b_init = tf.ones_initializer
    with tf.variable_scope(scope):
        layer['W'] = tf.get_variable(
            scope + '_w',
            [input_size, output_size],
            initializer=w_init)
        layer['b'] = tf.get_variable(
            scope + '_b', [output_size],
            initializer=b_init)
        if activation is None:
            layer['output'] = tf.nn.xw_plus_b(inputs, layer['W'], layer['b'])
        else:
            layer['output'] = activation(
                tf.nn.xw_plus_b(inputs, layer['W'], layer['b'])
            )
    return layer
