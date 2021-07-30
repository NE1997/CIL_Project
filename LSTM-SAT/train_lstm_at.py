#import codecs
##import datetime
import os
import pickle
import random
import re
#from sys import argv
#import sys
import time

from nltk.chunk.util import accuracy

from lstm_at import *
import numpy as np
import tensorflow.compat.v1 as tf




flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer(
    'batch_size', 128, 'the batch_size of the training procedure')  # 32 128
flags.DEFINE_float('lr', 0.001, 'the learning rate')    # @@@@ 0.0005-0.01
flags.DEFINE_boolean('use_emb', True, 'use embedding')
flags.DEFINE_boolean('load_senti', False, 'load sentiment score')
flags.DEFINE_integer('emdedding_dim', 200, 'embedding dim')
flags.DEFINE_integer('hidden_neural_size', 200, 'LSTM hidden neural size')
flags.DEFINE_integer('hidden_layer_num', 200, 'LSTM hidden layer num')
flags.DEFINE_string('input_path', './data_senti_batched_all.pkl', 'dataset path')
flags.DEFINE_integer(
    'max_len', 128, 'max_len of training sentence')    # @@@@ limit
flags.DEFINE_float('init_scale', 0.1, 'init scale')
flags.DEFINE_integer('class_num', 2, 'class num')
flags.DEFINE_float('keep_prob', 0.9, 'dropout rate')    # @@@@ 0.8-1.0 0.5
flags.DEFINE_integer('num_epoch', 15, 'num epoch')
flags.DEFINE_string('out_dir', os.path.abspath(
    os.path.join(os.path.curdir, "runs")), 'output directory')
flags.DEFINE_integer('check_point_every', 10, 'checkpoint every num epoch ')
flags.DEFINE_integer('random_batch_no', 17579, 'random batch number')  # @@@@

# transmission of senti score  senti' = (senti -mid) * sca
flags.DEFINE_float('polar_thres', 0, 'threshold for polarities')
flags.DEFINE_integer('mid', 0, 'mid')
flags.DEFINE_float('sca', 0.8, 'sca')

out_log = "./log" + str(FLAGS.sca) + "_" + str(FLAGS.batch_size) + "_" + str(FLAGS.random_batch_no) + \
    "_" + str(FLAGS.lr) + "_" + str(FLAGS.keep_prob) + \
    "_" + str(FLAGS.max_len) + "_" + str(FLAGS.num_epoch) + \
    "_" + str(FLAGS.emdedding_dim) + ".txt"
out_file = open(out_log, "w")


prediction_record = "./log/prediction_at.txt"
'''
true_predict_record = "./log/true_predict.txt"
alphas_record = "./log/alphas.txt"
alphas_org_record = "./log/alphas_org.txt"
prediction_record_f = open(prediction_record, "w")
true_predict_record_f = open(true_predict_record, "w")
alphas_record_f = open(alphas_record, "w")
alphas_org_record_f = open(alphas_org_record, "w")

out_vu = "./log/vu.txt"
vu_file = open(out_vu, "w")
'''


class Config(object):
    hidden_neural_size = FLAGS.hidden_neural_size
    use_emb = FLAGS.use_emb
    embed_dim = FLAGS.emdedding_dim
    hidden_layer_num = FLAGS.hidden_layer_num
    class_num = FLAGS.class_num
    keep_prob = FLAGS.keep_prob
    lr = FLAGS.lr
    batch_size = FLAGS.batch_size
    word_in_sen = FLAGS.max_len
    num_epoch = FLAGS.num_epoch
    out_dir = FLAGS.out_dir
    checkpoint_every = FLAGS.check_point_every


def evaluate(model, session, x_test_batch, senti_test, y_test_batch, global_steps=None, summary_writer=None):
    num = 0
    acc = 0
    r = 0
    pre = []
    true = []
    alph = []
    alph_org = []
    for index in range(0, len(x_test_batch)):
        feed_dict = {}
        feed_dict[model.input_x] = x_test_batch[index]
        feed_dict[model.input_y] = y_test_batch[index]
        feed_dict[model.senti_matrix] = senti_test[index]
        feed_dict[model.dropout_keep_prob] = FLAGS.keep_prob
        fetches = [model.predictions, model.loss, model.rmse,
                   model.accuracy, model.correct_pred, model.alphas, model.alphas_org]
        predictions, lose, rmse, accuracy, correct_pred, alphas, alphas_org = session.run(
            fetches, feed_dict)
        acc += accuracy
        r += rmse
        num += 1
        pre.append(predictions)
        true.append(correct_pred)
        alph.append(alphas)
        alph_org.append(alphas_org)

    accuracy = acc / float(num)
    rmse = r / float(num)
    return accuracy, rmse, pre, true, alph, alph_org


def run_epoch(model, session, x_train_batch, senti_train, y_train_batch, global_steps, train_op, train_summary_writer):
    begin = int(time.time())
    all_index = list(range(0, len(x_train_batch)))
    #print(len(all_index))
    random_index = random.sample(all_index, FLAGS.random_batch_no)
    total_acc = 0
    total_rmse = 0.0

    for index in random_index:
        feed_dict = {}
        feed_dict[model.input_x] = x_train_batch[index]
        feed_dict[model.input_y] = y_train_batch[index]
        feed_dict[model.senti_matrix] = senti_train[index]
        feed_dict[model.dropout_keep_prob] = FLAGS.keep_prob

        _, predictlabel, truelabel, predictions, lose, rmse, accuracy, vu = session.run(
            [train_op, model.predictlabel, model.truelabel, model.predictions, model.loss,
             model.rmse, model.accuracy, model.vu], feed_dict)
        total_acc += accuracy
        total_rmse += rmse
        global_steps += 1

        #[rows, cols] = vu.shape
        #for i in range(rows):
         #   for j in range(cols):
          #      vu_file.write(str(vu[i, j]) + "\n")

    batch_acc = total_acc / float(FLAGS.random_batch_no)
    batch_rmse = total_rmse / float(FLAGS.random_batch_no)
    print("the train accuracy is %f " % (batch_acc))
    print("the train rmse is %f " % (batch_rmse))
    end = int(time.time())
    print("training epoch takes %d seconds" % (end - begin))
    out_file.write("the train accuracy is " + str(batch_acc) + "\n")
    out_file.write("the train rmse is " + str(batch_rmse) + "\n")

    return global_steps


def run_valid(valid_model, session, global_steps, x_test_batch, senti_test, y_test_batch, dev_summary_writer):
    valid_acc, valid_rmse, valid_prediction, valid_correct_pred, valid_alphas, valid_alphas_org = evaluate(
        valid_model, session, x_test_batch, senti_test, y_test_batch, global_steps, dev_summary_writer)
    print("the valid accuracy is %f" % (valid_acc))
    print("the valid rmse is %f" % (valid_rmse))
    out_file.write("the valid accuracy is " + str(valid_acc) + "\n")
    out_file.write("the valid rmse is " + str(valid_rmse) + "\n")
    #np.savetxt(prediction_record, valid_prediction)
    #np.savetxt(true_predict_record, valid_correct_pred)
    #np.savetxt(alphas_record, valid_alphas)
    #np.savetxt(alphas_org_record, valid_alphas_org)


def modify_senti(senti_batch):
    [dim1, dim2, dim3] = senti_batch.shape
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                if (senti_batch[i, j, k] != 0):
                    senti_batch[i, j, k] = abs(senti_batch[i, j, k] - FLAGS.polar_thres) * \
                        FLAGS.sca + FLAGS.mid
                # print(element)
    return senti_batch


def modify_senti_noabs(senti_batch):
    #print(senti_batch.shape)
    [dim1, dim2, dim3] = senti_batch.shape
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                if (senti_batch[i, j, k] != 0):
                    senti_batch[i, j, k] = (senti_batch[i, j, k] - FLAGS.polar_thres) * \
                        FLAGS.sca + FLAGS.mid
                # print(element)
    return senti_batch


def train_step():

    print("loading the dataset...")
    print("cpu 0, sca 0.8")
    train_config = Config()
    eval_train_config = Config()

    finame = open(FLAGS.input_path, 'rb')

    [x_train_batch, y_train_batch, x_val_batch, y_val_batch, x_test_batch, y_test_batch, senti_train_batch, senti_val_batch, senti_test_batch, max_feature,
        embedding_weight] = pickle.load(finame)



    senti_train_batch = np.array(senti_train_batch)
    senti_val_batch = np.array(senti_val_batch)
    senti_test_batch = np.array(senti_test_batch)

    senti_train = modify_senti_noabs(senti_train_batch)
    senti_val = modify_senti_noabs(senti_val_batch)
    senti_test = modify_senti_noabs(senti_test_batch)

    print("begin training")

    gpu_train_config = tf.ConfigProto()
    gpu_train_config.gpu_options.allow_growth = False
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(
            -1 * FLAGS.init_scale, 1 * FLAGS.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = Model_lstm_att(train_config, max_feature,
                                   embed_weight=embedding_weight)



        global_step = tf.Variable(
            0, name="global_step", trainable=False)

        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
    
        grads_and_vars = optimizer.compute_gradients(model.rmse)

        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)


        train_summary_dir = os.path.join(
            train_config.out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, session.graph)


        dev_summary_dir = os.path.join(
            eval_train_config.out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(
            dev_summary_dir, session.graph)

        # add checkpoint
        checkpoint_dir = os.path.abspath(
            os.path.join(train_config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        tf.initialize_all_variables().run()
        global_steps = 1
        begin_time = int(time.time())

        for i in range(train_config.num_epoch):
            print("the %d epoch training..." % (i + 1))
            out_file.write("the " + str((i + 1)) + " epoch training...\n")

            global_steps = run_epoch(
                model, session, x_train_batch, senti_train, y_train_batch, global_steps, train_op, train_summary_writer)
            run_valid(model, session, global_steps,
                      x_val_batch, senti_val, y_val_batch, dev_summary_writer)
            if i % train_config.checkpoint_every == 0:
                path = saver.save(session, checkpoint_prefix, global_steps)
                print("Saved model chechpoint to{}\n".format(path))

        print("the train is finished")
        end_time = int(time.time())
        print("training takes %d seconds already\n" % (end_time - begin_time))
        test_accuracy, test_rmse, test_prediction, test_correct_pred, test_alphas, test_alphas_org = evaluate(
            model, session, x_test_batch, senti_test, y_test_batch)

        for item in test_prediction:
            np.savetxt(prediction_record_f, item)


        print("program end!")


def main(_):
    train_step()


if __name__ == "__main__":
    tf.app.run()
