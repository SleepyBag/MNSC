import tensorflow as tf
from tensorflow import constant as const
from colored import stylize, fg
from math import sqrt
import numpy as np
lookup = tf.nn.embedding_lookup


def var(name, shape, initializer):
    return tf.get_variable(name, shape=shape, initializer=initializer)


class MDHUAPA(object):

    def __init__(self, args):
        self.max_doc_len = args['max_doc_len']
        self.cls_cnt = args['cls_cnt']
        self.embedding = args['embedding']
        self.emb_dim = args['emb_dim']
        self.hidden_size = args['hidden_size']
        self.hidden_size = args['hidden_size']
        self.usr_cnt = args['usr_cnt']
        self.prd_cnt = args['prd_cnt']
        self.hidden_size = args['hidden_size']
        self.hidden_size = args['hidden_size']
        self.sen_hop_cnt = args['sen_hop_cnt']
        self.doc_hop_cnt = args['doc_hop_cnt']
        self.l2_rate = args['l2_rate']
        self.convert_flag = args['convert_flag']
        self.debug = args['debug']
        self.lambda1 = args['lambda1']
        self.lambda2 = args['lambda2']
        self.lambda3 = args['lambda3']

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
                # 'wrd_emb': tf.Variable(embedding, name='wrd_emb', dtype=tf.float32),
                'usr_emb': var('usr_emb', [self.usr_cnt, hsize], self.emb_initializer),
                'prd_emb': var('prd_emb', [self.prd_cnt, hsize], self.emb_initializer),
            }

        # for tensorboard
        if self.debug:
            tf.summary.histogram('usr_emb', self.embeddings['usr_emb'])
            tf.summary.histogram('prd_emb', self.embeddings['prd_emb'])

    def fold(self, x, sen_len):
        with tf.variable_scope('fold'):
            # !!!
            x = tf.reshape(x, [-1, self.max_doc_len // sen_len, sen_len], name='cur_wrd')
            aug_x = tf.manip.roll(x, shift=-1, axis=1)
            x = tf.concat([x, aug_x], axis=2, name='cur_wrd')
            sen_len = tf.reduce_sum(1 - tf.cast(tf.equal(x, 0), tf.int32), axis=2)
            # document length substracts 1 because the last sentence is led by zeors,
            # which may cause problems
            doc_len = tf.reduce_sum(1 - tf.cast(tf.equal(sen_len, 0), tf.int32), axis=1) - 1
            x = lookup(self.embeddings['wrd_emb'], x, name='cur_wrd_embedding')
        return x, sen_len, doc_len

    def mdhuapa(self, x, usr, prd, convert_flag):
        # !!! 
        folds = [15]
        logitsu, logitsp = [], []
        for i, max_sen_len in enumerate(folds):
            with tf.variable_scope('divide' + str(i)):
                max_doc_len = self.max_doc_len // max_sen_len
                logitu, logitp = self.dhuapa(x, max_sen_len, max_doc_len, usr, prd, convert_flag)
                logitsu.append(logitu)
                logitsp.append(logitp)

        logitsu = tf.concat(logitsu, axis=1)
        logitsp = tf.concat(logitsu, axis=1)

        with tf.variable_scope('result'):

            softmax_w = var('softmax_w', [self.hidden_size * 2 * len(folds), self.cls_cnt], self.weights_initializer)
            # softmax_wup = var('softmax_wup', [self.hidden_size * 2 * len(folds), self.cls_cnt], self.weights_initializer)

            softmax_wu = var('softmax_wu', [self.hidden_size * len(folds), self.cls_cnt], self.weights_initializer)
            softmax_wp = var('softmax_wp', [self.hidden_size * len(folds), self.cls_cnt], self.weights_initializer)

            softmax_b = var('softmax_b', [self.cls_cnt], self.biases_initializer)
            softmax_bu = var('softmax_bu', [self.cls_cnt], self.biases_initializer)
            softmax_bp = var('softmax_bp', [self.cls_cnt], self.biases_initializer)

            d_hatu = tf.matmul(logitsu, softmax_wu) + softmax_bu
            d_hatp = tf.matmul(logitsp, softmax_wp) + softmax_bp
            outputs = tf.concat([logitsu, logitsp], axis=1, name='dhuapa_output')
            d_hat = tf.matmul(outputs, softmax_w) + softmax_b
            # d_hat = tf.tanh(d_hat, name='d_hat')
        return d_hat, d_hatu, d_hatp

    def dhuapa(self, x, max_sen_len, max_doc_len, usr, prd, convert_flag):
        # self.inputs = tf.reshape(x, [-1, self.max_doc_len, self.emb_dim])
        # self.sen_len = tf.reshape(self.sen_len, [-1])
        x, sen_len, doc_len = self.fold(x, max_sen_len)
        # !!!
        max_sen_len *= 2

        outputs = []
        for scope, identity in zip(['user_block', 'product_block'], [usr, prd]):
            with tf.variable_scope(scope):
                outputs.append(self.dnsc(x, max_sen_len, max_doc_len, sen_len, doc_len,
                                         identity, convert_flag))

        return outputs[0], outputs[1]

    def dnsc(self, x, max_sen_len, max_doc_len, sen_len, doc_len, identity, convert_flag):
        x = tf.reshape(x, [-1, max_sen_len, self.hidden_size])
        sen_len = tf.reshape(sen_len, [-1])

        def lstm(inputs, sequence_length, hidden_size, scope):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(hidden_size // 2, forget_bias=0.,
                                                initializer=tf.contrib.layers.xavier_initializer()),
                cell_bw=tf.nn.rnn_cell.LSTMCell(hidden_size // 2, forget_bias=0.,
                                                initializer=tf.contrib.layers.xavier_initializer()),
                inputs=inputs,
                sequence_length=sequence_length,
                dtype=tf.float32,
                scope=scope
            )
            outputs = tf.concat(outputs, axis=2)
            return outputs, state

        def attention(v, wh, h, wi, i, b, doc_len, real_max_len):
            """
            wi, i are two lists where wu, wp and u, p are
            real_max_len is equal to max_len at the document layer
            """
            h_shape = h.shape
            # batch_size = h_shape[0]
            max_len = h_shape[1]
            hidden_size = h_shape[2]
            ans = []
            with tf.variable_scope('attention'):
                for twi, ti in zip(wi, i):
                    h = tf.reshape(h, [-1, hidden_size])
                    h = tf.matmul(h, wh)
                    e = tf.reshape(h + b, [-1, max_len, hidden_size])
                    e = e + tf.matmul(ti, twi)[:, None, :]
                    e = tf.tanh(e)
                    e = tf.reshape(e, [-1, hidden_size])
                    e = tf.reshape(tf.matmul(e, v), [-1, real_max_len])
                    e = tf.nn.softmax(e, name='attention_with_null_word')
                    mask = tf.sequence_mask(doc_len, real_max_len, dtype=tf.float32)
                    e = (e * mask)[:, None, :]
                    _sum = tf.reduce_sum(e, reduction_indices=2, keepdims=True) + 1e-9
                    e = tf.div(e, _sum, name='attention_without_null_word')
                    # e = tf.reshape(e, [-1, max_doc_len], name='attention_without_null_word')
                    ans.append(e)
            return ans

        def hop(scope, last, sentence, sentence_bkg,
                background, hop_args, attention_args, convert_flag):
            with tf.variable_scope(scope):
                new_background = [None] * len(background)
                sentence = tf.stop_gradient(sentence) \
                    if not last else sentence
                sentence_bkg = tf.stop_gradient(sentence_bkg) \
                    if not last else sentence_bkg
                attention_args['h'] = sentence
                alphas = attention(**attention_args)
                for i, alpha in enumerate(alphas):
                    new_background[i] = tf.matmul(alpha, sentence_bkg)
                    # new_background[i] = tf.matmul(alpha, sentence_bkg) \
                    #     if not last else tf.matmul(alpha, sentence)
                    new_background[i] = tf.reshape(new_background[i], [-1, self.hidden_size],
                                                   name='new_background' + str(i))
                    # if not last:
                    if 'w' in convert_flag:
                        background[i] = tf.matmul(background[i], hop_args['convert_w'][i])
                    if 'b' in convert_flag:
                        background[i] += hop_args['convert_b']
                    if 'a' in convert_flag:
                        background[i] = tf.tanh(background[i])
                    if 'z' in convert_flag:
                        z = tf.matmul(background[i], hop_args['wz_old']) + \
                            tf.matmul(new_background[i], hop_args['wz_new']) + \
                            hop_args['zb']
                        z = tf.nn.sigmoid(z)
                        new_background[i] = z * background[i] + (1 - z) * new_background[i]
                    if 'o' in convert_flag:
                        new_background[i] = background[i] + new_background[i]
            return new_background

        def create_args(identity, length, max_length, convert_flag):
            hop_args = {}
            attention_args = {'i': [identity], 'doc_len': length, 'real_max_len': max_length}
            attention_args['wh'] = var('wh', [self.hidden_size, self.hidden_size], self.weights_initializer)
            attention_args['wi'] = [var('w', [self.hidden_size, self.hidden_size], self.weights_initializer)]
            attention_args['v'] = var('v', [self.hidden_size, 1], self.weights_initializer)
            if 'w' in convert_flag:
                hop_args['convert_w'] = var('convert_w', [self.hidden_size, self.hidden_size], self.weights_initializer)
            if 'z' in convert_flag:
                hop_args['wz_old'] = var('wz_old', [self.hidden_size, self.hidden_size], self.weights_initializer)
                hop_args['wz_new'] = var('wz_new', [self.hidden_size, self.hidden_size], self.weights_initializer)
                hop_args['zb'] = var('zb', [self.hidden_size], self.biases_initializer)
            if 'b' in convert_flag:
                hop_args['convert_b'] = var('convert_b', [self.hidden_size], self.biases_initializer)
            attention_args['b'] = var('attention_b', [self.hidden_size], self.biases_initializer)

            return hop_args, attention_args

        with tf.variable_scope('sentence_layer'):
            # lstm_outputs, _state = lstm(x, sen_len, self.hidden_size, 'lstm')
            # lstm_outputs = tf.reshape(lstm_outputs, [-1, max_sen_len, self.hidden_size])
            lstm_bkg, _state = lstm(x, sen_len, self.hidden_size, 'lstm_bkg')
            lstm_bkg = tf.reshape(lstm_bkg, [-1, max_sen_len, self.hidden_size])
            lstm_outputs = lstm_bkg
            hop_args, attention_args = create_args(identity, sen_len, max_sen_len, convert_flag)

            sen_bkg = attention_args['i']
            for i, _ in enumerate(sen_bkg):
                sen_bkg[i] = tf.tile(sen_bkg[i][:, None, :], (1, max_doc_len, 1))
                sen_bkg[i] = tf.reshape(sen_bkg[i], (-1, self.hidden_size))
            for ihop in range(self.sen_hop_cnt):
                attention_args['i'] = sen_bkg
                sen_bkg = hop('hop' + str(ihop), ihop == self.sen_hop_cnt - 1, lstm_outputs, lstm_bkg,
                              sen_bkg, hop_args, attention_args, convert_flag)
        outputs = [tf.reshape(bkg, [-1, max_doc_len, self.hidden_size])
                   for bkg in sen_bkg]
        outputs = sum(outputs)

        with tf.variable_scope('document_layer'):
            hop_args, attention_args = create_args(identity, doc_len, max_doc_len, convert_flag)
            # lstm_outputs, _state = lstm(outputs, doc_len, self.hidden_size, 'lstm')
            lstm_bkg, _state = lstm(outputs, doc_len, self.hidden_size, 'lstm_bkg')
            lstm_outputs = lstm_bkg

            doc_bkg = attention_args['i']
            for ihop in range(self.doc_hop_cnt):
                attention_args['i'] = doc_bkg
                doc_bkg = hop('hop' + str(ihop), ihop == self.doc_hop_cnt - 1, lstm_outputs, lstm_bkg,
                              doc_bkg, hop_args, attention_args, convert_flag)
        outputs = tf.concat(values=doc_bkg, axis=1, name='outputs')

        return outputs

    def build(self, data_iter):
        # get the inputs
        with tf.variable_scope('inputs'):
            input_map = data_iter.get_next()
            usrid, prdid, input_x, input_y = (input_map['usr'], input_map['prd'],
                                              input_map['content'], input_map['rating'])

            usr = lookup(self.embeddings['usr_emb'], usrid, name='cur_usr_embedding')
            prd = lookup(self.embeddings['prd_emb'], prdid, name='cur_prd_embedding')

        # build the process of model
        d_hat, d_hatu, d_hatp = self.mdhuapa(input_x, usr, prd, self.convert_flag)
        prediction = tf.argmax(d_hat, 1, name='predictions')

        with tf.variable_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=d_hat,
                                                                   labels=tf.one_hot(input_y, self.cls_cnt))
            lossu = tf.nn.softmax_cross_entropy_with_logits_v2(logits=d_hatu,
                                                               labels=tf.one_hot(input_y, self.cls_cnt))
            lossp = tf.nn.softmax_cross_entropy_with_logits_v2(logits=d_hatp,
                                                               labels=tf.one_hot(input_y, self.cls_cnt))
            self.loss = self.lambda1 * self.loss + self.lambda2 * lossu + self.lambda3 * lossp
            # self.loss *= tf.sqrt(tf.abs(tf.cast(prediction, tf.float32) - tf.cast(input_y, tf.float32)) + 1.)
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
            info = 'best dev acc: %.3f, best test acc: %.3f, NEW best test RMSE: %.3f' % \
                (self.best_dev_acc, self.best_test_acc, self.best_test_rmse)
        return info

    def train(self, optimizer, global_step):
        grads_and_vars = optimizer.compute_gradients(self.loss)

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        return train_op
