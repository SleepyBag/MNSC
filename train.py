import ipdb
# -*- coding: utf-8 -*-
# author: Xue Qianming

import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
# from tensorflow.python import debug as tf_debug

import data
from dnsc import DNSC
# import matplotlib.pyplot as plt

params = {
    'debug_params': [('debug', False, 'Debug or not')],
    'data_params': [('n_class', 5, "Numbers of class"),
                    ('dataset', 'yelp13', "The dataset")],
    'model_chooing': [('model', 'nsc,', 'Model to train')],
    'model_hyperparam': [("embedding_dim", 200, "Dimensionality of character embedding"),
                         ("hidden_size", 200, "hidden_size"),
                         ('max_sen_len', 50, 'max number of tokens per sentence'),
                         ('max_doc_len', 40, 'max number of tokens per sentence'),
                         ("lr", .001, "Learning rate"),
                         ("l2_rate", 1e-5, "rate of l2 regularization"),
                         ("lambda1", .4, "rate of l2 regularization"),
                         ("lambda2", .3, "rate of l2 regularization"),
                         ("lambda3", .3, "rate of l2 regularization"),
                         ("bilstm", True, "use biLSTM or LSTM"),
                         ("hop_cnt", 3, "number of hops")],
    'training_params': [("batch_size", 100, "Batch Size"),
                        ("num_epochs", 1000, "Number of training epochs"),
                        ("evaluate_every", 200, "Evaluate model on dev set after this many steps"),
                        ("training_method", 'adam', 'Method chose to tune the weights')],
    'misc_params': [("allow_soft_placement", True, "Allow device soft device placement"),
                    ("log_device_placement", False, "Log placement of ops on devices")]
}

for param_collection in params.values():
    for param_name, default, description in param_collection:
        param_type = type(default)
        if param_type is int:
            tf.flags.DEFINE_integer(param_name, default, description)
        elif param_type is float:
            tf.flags.DEFINE_float(param_name, default, description)
        elif param_type is str:
            tf.flags.DEFINE_string(param_name, default, description)
        elif param_type is bool:
            tf.flags.DEFINE_boolean(param_name, default, description)

FLAGS = tf.flags.FLAGS
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{} = {}".format(attr, value.value))
# print("")


# Load data
print("Loading data...")
datasets = ['data/' + FLAGS.dataset + s for s in ['/train.ss', 'dev.ss', 'test.ss']]
embeddingpath = 'data/' + FLAGS.dataset + '/embedding.txt'
trainset, devset, testset = data.build_dataset(datasets, embeddingpath)
trainset = trainset.repeat().shuffle().batch(FLAGS.batch_size)
devset = devset.batch(FLAGS.batch_size)
testset = testset.batch(FLAGS.batch_size)
# trainset = Dataset('data/' + FLAGS.dataset + '/train.ss')
# devset = Dataset('data/' + FLAGS.dataset + '/dev.ss')
# testset = Dataset('data/' + FLAGS.dataset + '/test.ss')

# alldata = np.concatenate([trainset.t_docs, devset.t_docs, testset.t_docs], axis=0)
# embeddingfile, wordsdict, index_to_word = data_helpers.load_embedding(embeddingpath, alldata, FLAGS.embedding_dim)
# del alldata
print("Loading data finished...")

# usrdict, prddict = trainset.get_usr_prd_dict()
# trainbatches = trainset.batch_iter(usrdict, prddict, wordsdict, FLAGS.n_class, FLAGS.batch_size,
#                                    FLAGS.num_epochs, FLAGS.max_sen_len, FLAGS.max_doc_len)
# devset.genBatch(usrdict, prddict, wordsdict, FLAGS.batch_size,
#                 FLAGS.max_sen_len, FLAGS.max_doc_len, FLAGS.n_class)
# testset.genBatch(usrdict, prddict, wordsdict, FLAGS.batch_size,
#                  FLAGS.max_sen_len, FLAGS.max_doc_len, FLAGS.n_class)


def convert_index_to_words(indices):
    return [index_to_word[i] for i in indices]


with tf.Graph().as_default():
    # create the session
    session_config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)

    with sess.as_default():
        # build the model
        model_params = {
            'max_sen_len': FLAGS.max_sen_len,
            'max_doc_len': FLAGS.max_doc_len,
            'cls_cnt': FLAGS.n_class,
            'emb_file': embeddingfile,
            'emb_dim': FLAGS.embedding_dim,
            'hidden_size': FLAGS.hidden_size,
            'usr_cnt': len(usrdict),
            'prd_cnt': len(prddict),
            'l2_rate': FLAGS.l2_rate,
            'hop_cnt': FLAGS.hop_cnt
        }
        if FLAGS.model == 'dnsc':
            model = DNSC(**model_params)

        loss, mse, correct_num, accuracy = model.build()

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        if FLAGS.training_method == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr)
        elif FLAGS.training_method == 'adam':
            optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        elif FLAGS.training_method == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.lr, epsilon=1e-6)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Save dict
        timestamp = str(int(time.time()))
        checkpoint_dir = os.path.abspath("checkpoints/" + FLAGS.dataset + "/" + timestamp)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        with open(checkpoint_dir + "/wordsdict.txt", 'wb') as f:
            pickle.dump(wordsdict, f)
        with open(checkpoint_dir + "/usrdict.txt", 'wb') as f:
            pickle.dump(usrdict, f)
        with open(checkpoint_dir + "/prddict.txt", 'wb') as f:
            pickle.dump(prddict, f)

        sess.run(tf.global_variables_initializer())

        def train_step(batch, loss, accuracy):
            ipdb.set_trace()
            u, p, x, y, sen_len, doc_len = zip(*batch)
            feed_dict = {
                model.usrid: u,
                model.prdid: p,
                model.input_x: x,
                model.input_y: y,
                model.sen_len: sen_len,
                model.doc_len: doc_len
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, loss, accuracy],
                feed_dict)
            return step, loss, accuracy

        def predict_step(u, p, x, y, sen_len, doc_len, fetches, name=None):
            feed_dict = {
                model.usrid: u,
                model.prdid: p,
                model.input_x: x,
                model.input_y: y,
                model.sen_len: sen_len,
                model.doc_len: doc_len
            }
            fetches = sess.run(fetches, feed_dict)
            return fetches

        def predict(dataset, fetches, correct_num, mse, name=None):
            acc = 0
            rmse = 0.
            fetches = (correct_num, mse) + fetches
            pgb = tqdm(xrange(dataset.epoch), leave=False)
            for i in pgb:
                fetched = predict_step(dataset.usr[i], dataset.prd[i], dataset.docs[i],
                                       dataset.label[i], dataset.sen_len[i], dataset.doc_len[i],
                                       fetches, name)
                cur_correct_num, cur_mse = fetched[:2]
                acc += cur_correct_num
                rmse += cur_mse
            acc = acc * 1.0 / dataset.data_size
            rmse = np.sqrt(rmse / dataset.data_size)
            return acc, rmse

        topacc = 0.
        toprmse = 0.
        better_dev_acc = 0.
        predict_round = 0

        total_loss, total_accuracy = 0., 0.
        # Training loop. For each batch...
        for tr_batch in trainbatches:
            # pgb = tqdm(range(FLAGS.evaluate_every))
            # for i in pgb:
            cur_step, cur_loss, cur_accuracy = train_step(tr_batch, loss, accuracy)
            total_loss += cur_loss
            total_accuracy += cur_loss
            # pgb.set_description(message)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nstep %d, loss %.4f, acc %.4f" %
                      (cur_step, total_loss / FLAGS.evaluate_every, total_accuracy / FLAGS.evaluate_every))
                total_loss, total_accuracy = 0., 0.
                predict_round += 1
                print("Evaluation round %d:" % (predict_round))

                fetches = (loss, accuracy)
                dev_acc, dev_rmse = predict(devset, fetches, correct_num, mse, name="dev")
                print("dev_acc: %.4f    dev_RMSE: %.4f" % (dev_acc, dev_rmse))
                test_acc, test_rmse = predict(testset, fetches, correct_num, mse, name="test")
                print("test_acc: %.4f    test_RMSE: %.4f" % (test_acc, test_rmse))

                # print topacc with best dev acc
                if dev_acc >= better_dev_acc:
                    better_dev_acc = dev_acc
                    topacc = test_acc
                    toprmse = test_rmse
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                print("topacc: %.4f   RMSE: %.4f" % (topacc, toprmse))

                # if FLAGS.debug:
                #     doc = tr_batch[0][2].tolist()
                #     for sentence, att in zip(doc, cur_attention):
                #         att = ['%.2f' % i for i in att]
                #         if sum(sentence) != 0:
                #             sentence = [i for i in sentence if i != 0]
                #             words = convert_index_to_words(sentence)
                #             # plt.bar(range(len(words)), att, tick_label=words)
                #             print(tabulate([words, att[:len(words)]]))
