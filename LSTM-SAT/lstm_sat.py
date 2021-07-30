# -*- coding:utf-8 -*-


import os
import pickle
import sys



import numpy as np
import tensorflow.compat.v1 as tf


def attention(inputs, attention_size, senti_matrix, time_major=False, return_alphas=True):
    if isinstance(inputs, tuple):

        inputs = tf.concat(inputs, 2)

    if time_major:

        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2]

    W_omega = tf.Variable(tf.random_normal(
        [hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))


    v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega)

    vu = tf.tensordot(v, u_omega, axes=1)  # (B,T) shape
    alphas_org = tf.nn.softmax(vu)
    vu_senti = vu + senti_matrix
    alphas = tf.nn.softmax(vu_senti)  # (B,T) shape also

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output, vu
    else:
        return output, alphas, vu, alphas_org


def embedding(use_embed,
              emb_weight,
              vocab_size,
              wordinsent_cnt,
              embedding_size,
              input_x):
    if use_embed:

        embeddings = tf.Variable(emb_weight, dtype=tf.float32)

    else:
        embeddings1 = tf.Variable(
            tf.zeros([1, embedding_size]), dtype=tf.float32)
        embeddings2 = tf.Variable(tf.random_uniform(
            [vocab_size + 1, embedding_size], -0.1, 0.1), name="embeddings")
        embeddings = tf.concat([embeddings1, embeddings2], axis=0)
    embedded_words = tf.nn.embedding_lookup(embeddings, input_x)

    embedded_words = tf.reshape(
        embedded_words, [-1, wordinsent_cnt, embedding_size])

    return embedded_words


class Model_base():

    def __init__(self, Config, max_feature, embed_weight):
        #self.gpuind = Config.gpu_id
        self.embedding_size = Config.embed_dim
        self.vocab_size = max_feature
        self.wordinsent_cnt = Config.word_in_sen
        self.class_cnt = Config.class_num
        self.hidden_size = Config.hidden_layer_num
        self.use_embed = Config.use_emb
        self.attention_size = self.embedding_size

        self.input_x = tf.placeholder(tf.int32,
                                      [None, self.wordinsent_cnt],
                                      name="input_x")
        self.input_y = tf.placeholder(tf.float32,
                                      [None, self.class_cnt],
                                      name="input_y")
        self.senti_matrix = tf.placeholder(
            tf.float32, [None, self.wordinsent_cnt], name="senti_matrix")
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                name="dropout_keep_prob")


class Model_lstm_att(Model_base):

    def __init__(self, Config, max_feature, embed_weight):
        super(Model_lstm_att, self).__init__(Config, max_feature, embed_weight)

        # Layer 1: Word embeddings
        self.embedded_words = embedding(self.use_embed,
                                        embed_weight,
                                        self.vocab_size,
                                        self.wordinsent_cnt,
                                        self.embedding_size,
                                        self.input_x
                                        )

        self.embedded_words = tf.reshape(
            self.embedded_words, [-1, self.wordinsent_cnt, self.embedding_size])

        # Layer 2: LSTM cell
        lstm_use_peepholes = True
        with tf.variable_scope('lstm_cell1'):
            print("Using simple 1-layer LSTM with hidden layer size {0}."
                  .format(self.hidden_size))
            self.lstm_cells = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size,
                                                forget_bias=1.0,
                                                use_peepholes=lstm_use_peepholes)

            self.lstm_cells_dropout = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cells,
                                                              input_keep_prob=self.dropout_keep_prob,
                                                              output_keep_prob=self.dropout_keep_prob)

            self.outputs1, _states1 = tf.nn.dynamic_rnn(self.lstm_cells_dropout,
                                                  inputs=self.embedded_words,
                                                  dtype=tf.float32)


            self.outputs_pool, self.alphas, self.vu, self.alphas_org = attention(
                self.outputs1, self.attention_size, self.senti_matrix, False, True)

#         outputs = tf.reduce_mean(self.outputs1, axis=1)

        # Layer 3: Final Softmax
        self.out_weight = tf.Variable(tf.random_normal(
            [self.hidden_size, self.class_cnt]))
        self.out_bias = tf.Variable(tf.random_normal([self.class_cnt]))

        with tf.name_scope("output"):
            lstm_final_output = self.outputs_pool
            self.scores = tf.nn.xw_plus_b(lstm_final_output, self.out_weight,
                                          self.out_bias, name="scores")
            self.predictions = tf.nn.softmax(self.scores, name="predictions")

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                                  labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses, name="loss")

        with tf.name_scope("rmse"):
            self.rmses = tf.square(tf.subtract(self.scores, self.input_y))
            self.rmse = tf.sqrt(tf.reduce_mean(self.rmses), name="rmse")

        with tf.name_scope("accuracy"):
            self.predictlabel = tf.argmax(self.predictions, 1)
            self.truelabel = tf.argmax(self.input_y, 1)
            self.correct_pred = tf.equal(tf.argmax(self.predictions, 1),
                                         tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"),
                                           name="accuracy")
