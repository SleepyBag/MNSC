# -*- coding: utf-8 -*-
# author: Xue Qianming
import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import data
from dnsc import DNSC
from colored import fg, stylize
import dnsc

params = {
    'debug_params': [('debug', False, 'Whether to debug or not')],
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


with tf.Graph().as_default():
    # Load data
    print("Loading data...")
    datasets = ['data/' + FLAGS.dataset + s for s in ['/train.ss', '/dev.ss', '/test.ss']]
    embeddingpath = 'data/' + FLAGS.dataset + '/embedding.txt'
    datasets, lengths, embedding, usr_cnt, prd_cnt = data.build_dataset(datasets, embeddingpath)
    trainset, devset, testset = datasets
    trainlen, devlen, testlen = lengths
    trainset = trainset.repeat().shuffle(100000).batch(FLAGS.batch_size)
    devset = devset.batch(FLAGS.batch_size).repeat()
    testset = testset.batch(FLAGS.batch_size).repeat()
    print("Loading data finished...")

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
            'max_sen_len': FLAGS.max_sen_len, 'max_doc_len': FLAGS.max_doc_len,
            'cls_cnt': FLAGS.n_class, 'embedding': embedding,
            'emb_dim': FLAGS.embedding_dim, 'hidden_size': FLAGS.hidden_size,
            'usr_cnt': usr_cnt, 'prd_cnt': prd_cnt,
            'l2_rate': FLAGS.l2_rate, 'hop_cnt': FLAGS.hop_cnt
        }
        if FLAGS.model == 'dnsc':
            model = DNSC(**model_params)
            cur_model = dnsc

        data_iter = tf.data.Iterator.from_structure(trainset.output_types, output_shapes=trainset.output_shapes)
        traininit = data_iter.make_initializer(trainset)
        devinit = data_iter.make_initializer(devset)
        testinit = data_iter.make_initializer(testset)

        metrics = model.build(data_iter)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        if FLAGS.training_method == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr)
        elif FLAGS.training_method == 'adam':
            optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        elif FLAGS.training_method == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.lr, epsilon=1e-6)
        grads_and_vars = optimizer.compute_gradients(metrics[0])
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # merge tensorboard summary
        if FLAGS.debug:
            summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('summary/train', sess.graph)
            dev_writer = tf.summary.FileWriter('summary/dev', sess.graph)
            test_writer = tf.summary.FileWriter('summary/test', sess.graph)

        sess.run(tf.global_variables_initializer())

        # run a dataset
        def run_set(sess, testlen, metrics, ops=tuple()):
            pgb = tqdm(range(testlen // FLAGS.batch_size), leave=False, ncols=90)
            metrics_total = [0] * len(metrics)
            op_results = [[] for i in ops]
            for i in pgb:
                cur_metrics = sess.run(metrics + ops)
                for j in range(len(metrics)):
                    metrics_total[j] += cur_metrics[j]
                for j in range(len(ops)):
                    op_results[j].append(cur_metrics[len(metrics) + j])
            return [metrics_total] + op_results

        best_dev_acc = 0.
        best_test_acc = 0.
        for epoch in range(FLAGS.num_epochs):
            # train on transet
            sess.run(traininit)
            trainlen = FLAGS.batch_size * FLAGS.evaluate_every
            # when debugging, summary info is needed for tensorboard
            if FLAGS.debug:
                train_metrics, step, train_summary, _ = \
                    run_set(sess, trainlen, metrics, (global_step, summary, train_op))
            else:
                train_metrics, step, _ = \
                    run_set(sess, trainlen, metrics, (global_step, train_op, ))
            info = model.output_metrics(train_metrics, trainlen)
            print(stylize('Trainset:' + info, fg('yellow')))

            if FLAGS.debug:
                for i, s in enumerate(train_summary):
                    train_writer.add_summary(s, step - FLAGS.evaluate_every + i)

            # test on devset
            sess.run(devinit)
            dev_metrics, = run_set(sess, devlen, metrics)
            info = model.output_metrics(dev_metrics, devlen)
            print(stylize('Devset:  ' + info, fg('green')))

            # test on testset
            sess.run(testinit)
            test_metrics, = run_set(sess, testlen, metrics)
            info = model.output_metrics(test_metrics, testlen)
            print(stylize('Testset: ' + info, fg('red')))

            info = model.record_metrics(dev_metrics, test_metrics, devlen, testlen)
            print(stylize('Epoch %d finished, ' % epoch + info, fg('white')))
