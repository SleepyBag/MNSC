import ipdb

import tensorflow as tf
from tensorflow import constant as const
from colored import stylize, fg
lookup = tf.nn.embedding_lookup


class MLSTM(object):

    def __init__(self, args):
        self.max_sen_len = args['max_sen_len']
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

        # initializers for parameters
        weights_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.contrib.layers.xavier_initializer()
        emb_initializer = tf.initializers.zeros()

        def var(name, shape, initializer):
            return tf.get_variable(name, shape=shape, initializer=initializer)

        hsize = self.hidden_size
        # weights in the model
        with tf.variable_scope('weights'):
            self.weights = {
                'softmax_w': var('softmax_w', [hsize * 2, self.cls_cnt], weights_initializer),
                'softmax_wu': var('softmax_wu', [hsize, self.cls_cnt], weights_initializer),
                'softmax_wp': var('softmax_wp', [hsize, self.cls_cnt], weights_initializer),
            }
            for prefix in ['sen_', 'doc_']:
                for suffix in ['u', 'p']:
                    def fix(s):
                        return prefix + s + suffix
                    # self.weights[fix('bkg_w')] = var(fix('bkg_w'), [hsize, hsize], weights_initializer)
                    self.weights[fix('wh')] = var(fix('wh'), [hsize, hsize], weights_initializer)
                    self.weights[fix('w')] = var(fix('w'), [hsize, hsize], weights_initializer)
                    self.weights[fix('v')] = var(fix('v'), [hsize, 1], weights_initializer)
                    self.weights[fix('convert_w')] = var(fix('convert_w'), [hsize, hsize], weights_initializer)
                    self.weights[fix('wz_old')] = var(fix('wz_old'), [hsize, hsize], weights_initializer)
                    self.weights[fix('wz_new')] = var(fix('wz_new'), [hsize, hsize], weights_initializer)

        # biases in the model
        with tf.variable_scope('biases'):
            self.biases = {
                'softmax_b': var('softmax_b', [self.cls_cnt], biases_initializer),
                'softmax_bu': var('softmax_bu', [self.cls_cnt], biases_initializer),
                'softmax_bp': var('softmax_bp', [self.cls_cnt], biases_initializer),
            }
            for prefix in ['sen_', 'doc_']:
                for suffix in ['u', 'p']:
                    def fix(s):
                        return prefix + s + suffix
                    self.biases[fix('convert_b')] = var(fix('convert_b'), [hsize], biases_initializer)
                    self.biases[fix('attention_b')] = var(fix('attention_b'), [hsize], biases_initializer)
                    self.biases[fix('zb')] = var(fix('zb'), [hsize], biases_initializer)

        # embeddings in the model
        with tf.variable_scope('emb'):
            self.embeddings = {
                'wrd_emb': const(self.embedding, name='wrd_emb', dtype=tf.float32),
                # 'wrd_emb': tf.Variable(embedding, name='wrd_emb', dtype=tf.float32),
                'usr_emb': var('usr_emb', [self.usr_cnt, hsize], emb_initializer),
                'prd_emb': var('prd_emb', [self.prd_cnt, hsize], emb_initializer),
            }

        # for tensorboard
        if self.debug:
            tf.summary.histogram('usr_emb', self.embeddings['usr_emb'])
            tf.summary.histogram('prd_emb', self.embeddings['prd_emb'])
            tf.summary.histogram('sen_convert_wu', self.weights['sen_convert_wu'])
            tf.summary.histogram('sen_convert_wp', self.weights['sen_convert_wp'])
            tf.summary.histogram('sen_convert_wu', self.weights['sen_convert_wu'])
            tf.summary.histogram('sen_convert_wp', self.weights['sen_convert_wp'])

    def dhuapa(self, x, usr, prd, convert_flag):
        self.inputs = x
        # self.inputs = tf.reshape(x, [-1, self.max_sen_len, self.emb_dim])
        # self.sen_len = tf.reshape(self.sen_len, [-1])

        outputs = []
        for scope, suffix, identity in zip(['user_block', 'product_block'],
                                           ['u', 'p'], [usr, prd]):
            with tf.variable_scope(scope):
                sen_hop_args = {
                    'convert_w': [self.weights['sen_convert_w' + suffix]],
                    'convert_b': [self.biases['sen_convert_b' + suffix]],
                    'wz_old': self.weights['sen_wz_old' + suffix],
                    'wz_new': self.weights['sen_wz_new' + suffix],
                    'zb': self.biases['sen_zb' + suffix]
                }
                doc_hop_args = {
                    'convert_w': [self.weights['doc_convert_w' + suffix]],
                    'convert_b': [self.biases['doc_convert_b' + suffix]],
                    'wz_old': self.weights['doc_wz_old' + suffix],
                    'wz_new': self.weights['doc_wz_new' + suffix],
                    'zb': self.biases['doc_zb' + suffix]
                }
                sen_attention_args = {'v': self.weights['sen_v' + suffix],
                                      'wh': self.weights['sen_wh' + suffix],
                                      'wi': [self.weights['sen_w' + suffix]],
                                      'i': [identity],
                                      'b': self.biases['sen_attention_b' + suffix],
                                      'doc_len': self.sen_len,
                                      'real_max_len': self.max_sen_len}
                doc_attention_args = {'v': self.weights['doc_v' + suffix],
                                      'wh': self.weights['doc_wh' + suffix],
                                      'wi': [self.weights['doc_w' + suffix]],
                                      'i': [identity],
                                      'b': self.biases['doc_attention_b' + suffix],
                                      'doc_len': self.doc_len,
                                      'real_max_len': self.max_doc_len}
                outputs.append(self.mlstm(sen_hop_args, doc_hop_args,
                                          sen_attention_args, doc_attention_args, convert_flag))

        with tf.variable_scope('result'):
            d_hatu = tf.matmul(outputs[0], self.weights['softmax_wu']) + self.biases['softmax_bu']
            d_hatp = tf.matmul(outputs[1], self.weights['softmax_wp']) + self.biases['softmax_bp']
            outputs = tf.concat(outputs, axis=1, name='dhuapa_output')
            d_hat = tf.matmul(outputs, self.weights['softmax_w']) + self.biases['softmax_b']
            # d_hat = tf.tanh(d_hat, name='d_hat')
        return d_hat, d_hatu, d_hatp

    def mlstm(self, sen_hop_args, doc_hop_args,
              sen_attention_args, doc_attention_args, convert_flag):

        def lstm(inputs, sequence_length, hidden_size, scope):
            outputs, state = tf.nn.dynamic_rnn(
                cell=tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=0.,
                                             initializer=tf.contrib.layers.xavier_initializer()),
                inputs=inputs,
                sequence_length=sequence_length,
                dtype=tf.float32,
                scope=scope
            )
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
                    e = tf.tanh(e, name='attention_with_null_word')
                    e = tf.reshape(e, [-1, hidden_size])
                    e = tf.reshape(tf.matmul(e, v), [-1, real_max_len])
                    e = tf.nn.softmax(e)
                    mask = tf.sequence_mask(doc_len, real_max_len, dtype=tf.float32)
                    e = (e * mask)[:, None, :]
                    _sum = tf.reduce_sum(e, reduction_indices=2, keepdims=True) + 1e-9
                    e = e / _sum
                    e = tf.identity(e, name='attention_without_null_word')
                    # e = tf.reshape(e, [-1, max_doc_len], name='attention_without_null_word')
                    ans.append(e)
            return ans

        def hop(scope, last, sentence, sentence_shape, attention_shape,
                background, hop_args, attention_args, convert_flag):
            with tf.variable_scope(scope):
                new_background = [None] * len(background)
                sentence = tf.stop_gradient(sentence) \
                    if not last else sentence
                if attention_shape is not None:
                    sentence = tf.reshape(sentence, attention_shape)
                attention_args['h'] = sentence
                alphas = attention(**attention_args)
                if attention_shape is not None:
                    sentence = tf.reshape(sentence, sentence_shape)
                for i, alpha in enumerate(alphas):
                    ipdb.set_trace()
                    new_background[i] = tf.matmul(alpha, sentence)
                    new_background[i] = tf.reshape(new_background[i], [-1, self.hidden_size],
                                                   name='new_background' + str(i))
                    if not last:
                        if 'w' in convert_flag:
                            background[i] = tf.matmul(background[i], hop_args['convert_w'][i])
                        if 'b' in convert_flag:
                            background[i] += hop_args['convert_b']
                        # activation func!!!
                        # if 'w' in convert_flag or 'b' in convert_flag:
                        #     background[i] = tf.tanh(background[i])
                        if 'z' in convert_flag:
                            z = tf.matmul(background[i], hop_args['wz_old']) + \
                                tf.matmul(new_background[i], hop_args['wz_new']) + \
                                hop_args['zb']
                            z = tf.nn.sigmoid(z)
                            new_background[i] = z * background[i] + (1 - z) * new_background[i]
                        if 'o' in convert_flag:
                            new_background[i] = background[i] + new_background[i]
            return new_background

        lstm_outputs, _state = lstm(self.inputs, self.sen_len, self.hidden_size, 'lstm')
        # convert_w = [self.weights['sen_convert_wu'], self.weights['sen_convert_wp']]
        # convert_b = [self.biases['sen_convert_bu'], self.biases['sen_convert_bp']]

        sen_bkg = sen_attention_args['i']
        for ihop in range(self.sen_hop_cnt):
            sen_attention_args['i'] = sen_bkg
            attention_shape, sentence_shape = None, None
            # attention_shape = [-1, self.max_doc_len * self.max_sen_len, self.hidden_size]
            # sentence_shape = [-1, self.max_sen_len, self.hidden_size]
            outputs = hop('hop' + str(ihop), ihop == self.sen_hop_cnt - 1, lstm_outputs,
                          sentence_shape, attention_shape, sen_bkg,
                          sen_hop_args, sen_attention_args, convert_flag)
            # outputs = [tf.reshape(bkg, [-1, self.max_doc_len, self.hidden_size])
            #            for bkg in outputs]

        outputs = tf.concat(values=outputs, axis=1, name='outputs')

        return outputs

    def build(self, data_iter):
        # get the inputs
        with tf.variable_scope('inputs'):
            input_map = data_iter.get_next()
            usrid, prdid, input_x, input_y, self.doc_len, self.sen_len = \
                (input_map['usr'], input_map['prd'], input_map['content'],
                 input_map['rating'], input_map['doc_len'], input_map['sen_len'])

            x = lookup(self.embeddings['wrd_emb'], input_x, name='cur_wrd_embedding')
            usr = lookup(self.embeddings['usr_emb'], usrid, name='cur_usr_embedding')
            prd = lookup(self.embeddings['prd_emb'], prdid, name='cur_prd_embedding')

        # build the process of model
        d_hat, d_hatu, d_hatp = self.dhuapa(x, usr, prd, self.convert_flag)
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
        info = 'Loss = %.3f, MSE = %.3f, Acc = %.3f' % \
            (loss / data_length, float(mse) / data_length, float(correct_num) / data_length)
        return info

    def record_metrics(self, dev_metrics, test_metrics, devlen, testlen):
        _dev_loss, _dev_mse, dev_correct_num, dev_accuracy = dev_metrics
        _test_loss, _test_mse, test_correct_num, test_accuracy = test_metrics
        dev_accuracy = float(dev_correct_num) / devlen
        test_accuracy = float(test_correct_num) / testlen
        if dev_accuracy > self.best_dev_acc:
            self.best_dev_acc = dev_accuracy
            self.best_test_acc = test_accuracy
            info = 'NEW best dev acc: %.3f, NEW best test acc: %.3f' % \
                (self.best_dev_acc, self.best_test_acc)
        else:
            info = 'best dev acc: %.3f, best test acc: %.3f' % \
                (self.best_dev_acc, self.best_test_acc)
        return info

    def train(self, optimizer, global_step):
        grads_and_vars = optimizer.compute_gradients(self.loss)
        # for g, v in grads_and_vars:
        #     if v is self.weights['']:
        #         pass
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        return train_op
