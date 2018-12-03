import ipdb
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
from dnsc2 import DNSC2
from colored import fg, stylize
import dnsc
import dnsc2
import math


params = {
    'debug_params': [('debug', False, 'Whether to debug or not')],
    'data_params': [('cls_cnt', 10, "Numbers of class"),
                    ('dataset', 'test', "The dataset")],
    'model_chooing': [('model', 'dnsc', 'Model to train')],
    'model_hyperparam': [("emb_dim", 200, "Dimensionality of character embedding"),
                         ("hidden_size", 200, "hidden_size"),
                         ('max_sen_len', 50, 'max number of tokens per sentence'),
                         ('max_doc_len', 40, 'max number of tokens per sentence'),
                         ("lr", .001, "Learning rate"),
                         ("l2_rate", .001, "rate of l2 regularization"),
                         ("lambda1", .4, "rate of l2 regularization"),
                         ("lambda2", .3, "rate of l2 regularization"),
                         ("lambda3", .3, "rate of l2 regularization"),
                         ("bilstm", True, "use biLSTM or LSTM"),
                         ("sen_hop_cnt", 1, "number of hops in sentence layer"),
                         ("doc_hop_cnt", 1, "number of hops in document layer"),
                         ("convert_flag", 'obw', "params used in background converting process")],
    'training_params': [("batch_size", 100, "Batch Size"),
                        ("epoch_cnt", 1000, "Number of training epochs"),
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

flags = tf.flags.FLAGS

# save current codes
cur_time = time.time()
os.system('mkdir code_history/' + str(cur_time))
os.system('cp *.py code_history/' + str(cur_time) + '/')
localtime = time.localtime(cur_time)

_ = flags.batch_size
# print params
output_file = open('code_history/' + str(cur_time) + '/output.txt', 'a')
print("\nParameters:")
for attr, value in sorted(flags.__flags.items()):
    print("{}={}".format(attr.upper(), value.value))
    print >> output_file, "{}={}".format(attr.upper(), value.value)
print("")
output_file.close()


with tf.Graph().as_default():
    # Load data
    print("Loading data...")
    datasets = ['data/' + flags.dataset + s for s in ['/train.ss', '/dev.ss', '/test.ss']]
    embeddingpath = 'data/' + flags.dataset + '/embedding.txt'
    datasets, lengths, embedding, usr_cnt, prd_cnt = \
        data.build_dataset(datasets, embeddingpath, flags.max_doc_len, flags.max_sen_len)
    trainset, devset, testset = datasets
    trainlen, devlen, testlen = lengths
    trainset = trainset.shuffle(100000).batch(flags.batch_size)
    devset = devset.batch(flags.batch_size)
    testset = testset.batch(flags.batch_size)
    print("Loading data finished...")

    # create the session
    session_config = tf.ConfigProto(
        allow_soft_placement=flags.allow_soft_placement,
        log_device_placement=flags.log_device_placement
    )
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)

    with sess.as_default():
        # build the model
        # model_params = flags.__flags
        # model_params['embedding'] = embedding
        # model_params['usr_cnt'] = usr_cnt
        # model_params['prd_cnt'] = prd_cnt
        model_params = {
            'max_sen_len': flags.max_sen_len, 'max_doc_len': flags.max_doc_len,
            'cls_cnt': flags.cls_cnt, 'embedding': embedding,
            'emb_dim': flags.emb_dim, 'hidden_size': flags.hidden_size,
            'usr_cnt': usr_cnt, 'prd_cnt': prd_cnt,
            'l2_rate': flags.l2_rate, 'sen_hop_cnt': flags.sen_hop_cnt,
            'doc_hop_cnt': flags.doc_hop_cnt,
            'debug': flags.debug, 'convert_flag': flags.convert_flag,
            'lambda1': flags.lambda1, 'lambda2': flags.lambda2, 'lambda3': flags.lambda3
        }
        if flags.model == 'dnsc':
            model = DNSC(model_params)
            cur_model = dnsc
        elif flags.model == 'dnsc2':
            model = DNSC2(model_params)
            cur_model = dnsc2

        data_iter = tf.data.Iterator.from_structure(trainset.output_types,
                                                    output_shapes=trainset.output_shapes)
        traininit = data_iter.make_initializer(trainset)
        devinit = data_iter.make_initializer(devset)
        testinit = data_iter.make_initializer(testset)

        metrics = model.build(data_iter)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        if flags.training_method == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(flags.lr)
        elif flags.training_method == 'adam':
            optimizer = tf.train.AdamOptimizer(flags.lr)
        elif flags.training_method == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(flags.lr, epsilon=1e-6)
        train_op = model.train(optimizer, global_step)

        # merge tensorboard summary
        if flags.debug:
            summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('summary/train', sess.graph)
            # dev_writer = tf.summary.FileWriter('summary/dev', sess.graph)
            # test_writer = tf.summary.FileWriter('summary/test', sess.graph)

        sess.run(tf.global_variables_initializer())

        # run a dataset
        def run_set(sess, testlen, metrics, ops=tuple()):
            pgb = tqdm(range(int(math.ceil(float(testlen) / flags.batch_size))),
                       leave=False, ncols=50)
            metrics_total = [0] * len(metrics)
            op_results = [[] for i in ops]
            for i in pgb:
                cur_metrics = sess.run(metrics + ops)
                for j in range(len(metrics)):
                    metrics_total[j] += cur_metrics[j]
                for j in range(len(ops)):
                    op_results[j].append(cur_metrics[len(metrics) + j])
            return [metrics_total] + op_results

        try:
            for epoch in range(flags.epoch_cnt):
                sess.run(traininit)
                output_file = open('code_history/' + str(cur_time) + '/output.txt', 'a')
                # train on transet
                # trainlen = flags.batch_size * flags.evaluate_every
                # when debugging, summary info is needed for tensorboard
                if flags.debug:
                    train_metrics, step, train_summary, _ = \
                        run_set(sess, trainlen, metrics, (global_step, summary, train_op))
                else:
                    train_metrics, step, _ = \
                        run_set(sess, trainlen, metrics, (global_step, train_op, ))
                info = model.output_metrics(train_metrics, trainlen)
                info = 'Trainset:' + info
                print(stylize(info, fg('yellow')))
                print >> output_file, info

                if flags.debug:
                    for i, s in zip(step, train_summary):
                        train_writer.add_summary(s, i)

                # test on devset
                sess.run(devinit)
                dev_metrics, = run_set(sess, devlen, metrics)
                info = model.output_metrics(dev_metrics, devlen)
                info = 'Devset:  ' + info
                print(stylize(info, fg('green')))
                print >> output_file, info

                # test on testset
                sess.run(testinit)
                test_metrics, = run_set(sess, testlen, metrics)
                info = model.output_metrics(test_metrics, testlen)
                info = 'Testset: ' + info
                print(stylize(info, fg('red')))
                print >> output_file, info

                info = model.record_metrics(dev_metrics, test_metrics, devlen, testlen)
                info = 'Epoch %d finished, ' % epoch + info
                print(stylize(info, fg('white')))
                print >> output_file, info

                output_file.close()

        except KeyboardInterrupt:
            print('Interrupted')
            best_test_acc = model.best_test_acc
            src = 'code_history/' + str(cur_time)
            dest = 'code_history/' + 'acc' + str(best_test_acc) + '_' + str(cur_time)
            os.system('mv ' + src + ' ' + dest)
