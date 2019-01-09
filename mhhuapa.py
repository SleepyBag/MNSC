import tensorflow as tf
from tensorflow import constant as const
from hop import hop
from colored import stylize, fg
from math import sqrt
import numpy as np
lookup = tf.nn.embedding_lookup


def var(name, shape, initializer):
    return tf.get_variable(name, shape=shape, initializer=initializer)


class MHHUAPA(object):

    def __init__(self, args):
        self.max_doc_len = args['max_doc_len']
        self.max_sen_len = args['max_sen_len']
        self.cls_cnt = args['cls_cnt']
        self.embedding = args['embedding']
        self.emb_dim = args['emb_dim']
        self.hidden_size = args['hidden_size']
        self.usr_cnt = args['usr_cnt']
        self.prd_cnt = args['prd_cnt']
        self.sen_hop_cnt = args['sen_hop_cnt']
        self.doc_hop_cnt = args['doc_hop_cnt']
        self.l2_rate = args['l2_rate']
        self.convert_flag = args['convert_flag']
        self.debug = args['debug']
        self.lambda1 = args['lambda1']
        self.lambda2 = args['lambda2']
        self.lambda3 = args['lambda3']
        self.head_cnt = 3

        self.best_dev_acc = .0
        self.best_test_acc = .0
        self.best_test_rmse = .0

        # initializers for parameters
        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.biases_initializer = tf.initializers.zeros()
        self.emb_initializer = tf.contrib.layers.xavier_initializer()

        hsize = self.hidden_size

        # embeddings in the model
        with tf.variable_scope('emb'):
            self.embeddings = {
                'wrd_emb': const(self.embedding, name='wrd_emb', dtype=tf.float32),
                # 'wrd_emb': tf.Variable(self.embedding, name='wrd_emb', dtype=tf.float32),
                'usr_emb': var('usr_emb', [self.usr_cnt, hsize], self.emb_initializer),
                'prd_emb': var('prd_emb', [self.prd_cnt, hsize], self.emb_initializer)
            }

        # for tensorboard
        if self.debug:
            tf.summary.histogram('usr_emb', self.embeddings['usr_emb'])
            tf.summary.histogram('prd_emb', self.embeddings['prd_emb'])

    def dhuapa(self, x, max_sen_len, max_doc_len, sen_len, doc_len, usr, prd, convert_flag):
        logits, d_hats = [], []
        for scope, identities in zip(['user_block', 'product_block'], [[usr], [prd]]):
            with tf.variable_scope(scope):
                logit = self.dnsc(x, max_sen_len, max_doc_len, sen_len, doc_len,
                                  identities, convert_flag)
                logits.append(logit)

                with tf.variable_scope('result'):
                    d_hats.append(tf.layers.dense(logit, self.cls_cnt,
                                                  kernel_initializer=self.weights_initializer,
                                                  bias_initializer=self.biases_initializer))

        with tf.variable_scope('result'):
            logits = tf.concat(logits, axis=1, name='dhuapa_output')
            d_hat = tf.layers.dense(logits, self.cls_cnt,
                                    kernel_initializer=self.weights_initializer,
                                    bias_initializer=self.biases_initializer)

        return [d_hat] + d_hats

    def dnsc(self, x, max_sen_len, max_doc_len, sen_len, doc_len, identities, convert_flag):
        x = tf.reshape(x, [-1, max_sen_len, self.hidden_size])
        sen_len = tf.reshape(sen_len, [-1])

        def lstm(inputs, sequence_length, hidden_size, scope):
            cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_size // 2, forget_bias=0.,
                                              initializer=tf.contrib.layers.xavier_initializer())
            cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_size // 2, forget_bias=0.,
                                              initializer=tf.contrib.layers.xavier_initializer())
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=inputs,
                sequence_length=sequence_length, dtype=tf.float32, scope=scope)
            outputs = tf.concat(outputs, axis=2)
            return outputs, state

        with tf.variable_scope('sentence_layer'):
            # lstm_outputs, _state = lstm(x, sen_len, self.hidden_size, 'lstm')
            # lstm_outputs = tf.reshape(lstm_outputs, [-1, max_sen_len, self.hidden_size])
            lstm_bkg, _state = lstm(x, sen_len, self.hidden_size, 'lstm_bkg')
            lstm_bkg = tf.reshape(lstm_bkg, [-1, max_sen_len, self.hidden_size])
            lstm_outputs = lstm_bkg

            sen_bkg = [tf.reshape(tf.tile(bkg[:, None, :], (1, max_doc_len, 1)),
                                  (-1, self.hidden_size)) for bkg in identities]
            for ihop in range(self.sen_hop_cnt):
                last = ihop == self.sen_hop_cnt - 1
                sen_bkg[0] = hop('hop' + str(ihop), last, lstm_outputs, lstm_bkg,
                                 sen_bkg[0], sen_bkg[1:], sen_len, max_sen_len,
                                 convert_flag)
        outputs = tf.reshape(sen_bkg[0], [-1, max_doc_len, self.hidden_size])

        with tf.variable_scope('document_layer'):
            # lstm_outputs, _state = lstm(outputs, doc_len, self.hidden_size, 'lstm')
            lstm_bkg, _state = lstm(outputs, doc_len, self.hidden_size, 'lstm_bkg')
            lstm_outputs = lstm_bkg

            doc_bkg = [i for i in identities]

            outputs = [hop('hop' + str(i), last, lstm_outputs, lstm_bkg,
                           doc_bkg[0], doc_bkg[1:], doc_len, max_doc_len,
                           convert_flag) for i in range(self.head_cnt)]
        outputs = tf.concat(outputs, axis=1)

        return outputs

    def build(self, data_iter):
        # get the inputs
        with tf.variable_scope('inputs'):
            input_map = data_iter.get_next()
            usrid, prdid, input_x, input_y, sen_len, doc_len = \
                (input_map['usr'], input_map['prd'],
                 input_map['content'], input_map['rating'],
                 input_map['sen_len'], input_map['doc_len'])

            usr = lookup(self.embeddings['usr_emb'], usrid, name='cur_usr_embedding')
            prd = lookup(self.embeddings['prd_emb'], prdid, name='cur_prd_embedding')
            input_x = lookup(self.embeddings['wrd_emb'], input_x, name='cur_wrd_embedding')

        # build the process of model
        d_hat, d_hatu, d_hatp = self.dhuapa(input_x, self.max_sen_len, self.max_doc_len,
                                            sen_len, doc_len, usr, prd, self.convert_flag)
        prediction = tf.argmax(d_hat, 1, name='prediction')

        with tf.variable_scope("loss"):
            sce = tf.nn.softmax_cross_entropy_with_logits_v2
            self.loss = sce(logits=d_hat, labels=tf.one_hot(input_y, self.cls_cnt))
            lossu = sce(logits=d_hatu, labels=tf.one_hot(input_y, self.cls_cnt))
            lossp = sce(logits=d_hatp, labels=tf.one_hot(input_y, self.cls_cnt))

            self.loss = self.lambda1 * self.loss + self.lambda2 * lossu + self.lambda3 * lossp
            regularizer = tf.zeros(1)
            params = tf.trainable_variables()
            for param in params:
                if param not in self.embeddings.values():
                    regularizer += tf.nn.l2_loss(param)
            self.loss = tf.reduce_sum(self.loss) + self.l2_rate * regularizer

        with tf.variable_scope("metrics"):
            correct_prediction = tf.equal(prediction, input_y)
            mse = tf.reduce_sum(tf.square(prediction - input_y), name="mse")
            correct_num = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.int32), name="correct_num")
            accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"), name="accuracy")

        return self.loss, mse, correct_num, accuracy

    def output_metrics(self, metrics, data_length):
        loss, mse, correct_num, accuracy = metrics
        info = 'Loss = %.3f, RMSE = %.3f, Acc = %.3f' % \
            (loss / data_length, sqrt(float(mse) / data_length), float(correct_num) / data_length)
        return info

    def record_metrics(self, dev_metrics, test_metrics, devlen, testlen):
        _dev_loss, dev_mse, dev_correct_num, dev_accuracy = dev_metrics
        _test_loss, test_mse, test_correct_num, test_accuracy = test_metrics
        dev_accuracy = float(dev_correct_num) / devlen
        test_accuracy = float(test_correct_num) / testlen
        test_rmse = sqrt(float(test_mse) / testlen)
        if dev_accuracy > self.best_dev_acc:
            self.best_dev_acc = dev_accuracy
            self.best_test_acc = test_accuracy
            self.best_test_rmse = test_rmse
            info = 'NEW best dev acc: %.3f, NEW best test acc: %.3f, NEW best test RMSE: %.3f' % \
                (self.best_dev_acc, self.best_test_acc, self.best_test_rmse)
        else:
            info = 'best dev acc: %.3f, best test acc: %.3f, best test RMSE: %.3f' % \
                (self.best_dev_acc, self.best_test_acc, self.best_test_rmse)
        return info

    def train(self, optimizer, global_step):
        grads_and_vars = optimizer.compute_gradients(self.loss)

        # capped_grads_and_vars = []
        # for grad, var in grads_and_vars:
        #     if var is self.embeddings['wrd_emb']:
        #         grad = tf.IndexedSlices(grad.values * 0 * tf.cast(global_step > 3000, tf.float32),
        #                                 grad.indices, grad.dense_shape)
        #     capped_grads_and_vars.append((grad, var))

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        return train_op
