import tensorflow as tf
from tensorflow import constant as const
from colored import stylize, fg
lookup = tf.nn.embedding_lookup


def var(name, shape, initializer):
    return tf.get_variable(name, shape=shape, initializer=initializer)


class MLDHUAPA(object):

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
        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.biases_initializer = tf.contrib.layers.xavier_initializer()
        self.emb_initializer = tf.initializers.zeros()

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
            # tf.summary.histogram('sen_convert_wu', self.weights['sen_convert_wu'])
            # tf.summary.histogram('sen_convert_wp', self.weights['sen_convert_wp'])
            # tf.summary.histogram('sen_convert_wu', self.weights['sen_convert_wu'])
            # tf.summary.histogram('sen_convert_wp', self.weights['sen_convert_wp'])

    def dhuapa(self, x, usr, prd, convert_flag):
        self.inputs = tf.reshape(x, [-1, self.max_sen_len, self.emb_dim])
        self.sen_len = tf.reshape(self.sen_len, [-1])

        outputs = []
        for scope, suffix, identity in zip(['user_block', 'product_block'],
                                           ['u', 'p'], [usr, prd]):
            with tf.variable_scope(scope):
                outputs.append(self.mldhuapa(x, identity, convert_flag))

        with tf.variable_scope('result'):

            softmax_w = var('softmax_w', [self.hidden_size * 2, self.cls_cnt], self.weights_initializer)
            softmax_wu = var('softmax_wu', [self.hidden_size, self.cls_cnt], self.weights_initializer)
            softmax_wp = var('softmax_wp', [self.hidden_size, self.cls_cnt], self.weights_initializer)

            softmax_b = var('softmax_b', [self.cls_cnt], self.biases_initializer)
            softmax_bu = var('softmax_bu', [self.cls_cnt], self.biases_initializer)
            softmax_bp = var('softmax_bp', [self.cls_cnt], self.biases_initializer)

            d_hatu = tf.matmul(outputs[0], softmax_wu) + softmax_bu
            d_hatp = tf.matmul(outputs[1], softmax_wp) + softmax_bp
            outputs = tf.concat(outputs, axis=1, name='dhuapa_output')
            d_hat = tf.matmul(outputs, softmax_w) + softmax_b
            # d_hat = tf.tanh(d_hat, name='d_hat')
        return d_hat, d_hatu, d_hatp

    def mldhuapa(self, x, identity, convert_flag):

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
            # outputs, state = tf.nn.dynamic_rnn(
            #     cell=tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=0.,
            #                                  initializer=tf.contrib.layers.xavier_initializer()),
            #     inputs=inputs,
            #     sequence_length=sequence_length,
            #     dtype=tf.float32,
            #     scope=scope
            # )
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
                    new_background[i] = tf.matmul(alpha, sentence)
                    new_background[i] = tf.reshape(new_background[i], [-1, self.hidden_size],
                                                   name='new_background' + str(i))
                    if not last:
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

        def create_args(identity, length, max_length):
            wh = var('wh', [self.hidden_size, self.hidden_size], self.weights_initializer)
            w = var('w', [self.hidden_size, self.hidden_size], self.weights_initializer)
            v = var('v', [self.hidden_size, 1], self.weights_initializer)
            convert_w = var('convert_w', [self.hidden_size, self.hidden_size], self.weights_initializer)
            wz_old = var('wz_old', [self.hidden_size, self.hidden_size], self.weights_initializer)
            wz_new = var('wz_new', [self.hidden_size, self.hidden_size], self.weights_initializer)

            convert_b = var('convert_b', [self.hidden_size], self.biases_initializer)
            attention_b = var('attention_b', [self.hidden_size], self.biases_initializer)
            zb = var('zb', [self.hidden_size], self.biases_initializer)

            hop_args = {
                'convert_w': [convert_w],
                'convert_b': [convert_b],
                'wz_old': wz_old,
                'wz_new': wz_new,
                'zb': zb}
            attention_args = {'v': v,
                              'wh': wh,
                              'wi': [w],
                              'i': [identity],
                              'b': [attention_b],
                              'doc_len': length,
                              'real_max_len': max_length}
            return hop_args, attention_args

        def one_layer(self, x, identity, max_sen_len, sen_len, tile_cnt, convert_flag):
            with tf.variable_scope('sentence_layer'):
                x = tf.reshape(x, [-1, max_sen_len, self.hidden_size])
                sen_len = tf.reshape(sen_len, [-1])

                lstm_outputs, _state = lstm(x, sen_len, self.hidden_size, 'lstm')
                lstm_outputs = tf.reshape(lstm_outputs, [-1, max_sen_len, self.hidden_size])
                # convert_w = [self.weights['sen_convert_wu'], self.weights['sen_convert_wp']]
                # convert_b = [self.biases['sen_convert_bu'], self.biases['sen_convert_bp']]
                hop_args, attention_args = create_args(identity, sen_len, max_sen_len)

                sen_bkg = attention_args['i']
                for i, _ in enumerate(sen_bkg):
                    sen_bkg[i] = tf.tile(sen_bkg[i][:, None, :], (1, tile_cnt, 1))
                    sen_bkg[i] = tf.reshape(sen_bkg[i], (-1, self.hidden_size))
                for ihop in range(self.sen_hop_cnt):
                    attention_args['i'] = sen_bkg
                    attention_shape = None
                    sentence_shape = None
                    # attention_shape = [-1, self.max_doc_len * self.max_sen_len, self.hidden_size]
                    # sentence_shape = [-1, self.max_sen_len, self.hidden_size]
                    sen_bkg = hop('hop' + str(ihop), ihop == self.sen_hop_cnt - 1, lstm_outputs,
                                  sentence_shape, attention_shape, sen_bkg,
                                  hop_args, attention_args, convert_flag)
            # outputs = [tf.reshape(bkg, [-1, 10, self.hidden_size])
            #            for bkg in sen_bkg]
            outputs = sum(sen_bkg)
            return outputs

        sen_len = 1 - tf.cast(tf.equal(x, 0), tf.int32)
        x = lookup(self.embeddings['wrd_emb'], x, name='cur_wrd_embedding')
        max_doc_len = self.max_doc_len
        # max_sen_len = 10

        layer_cnt = 2
        max_sen_lens = [8, 5]
        for layer, max_sen_len in zip(range(layer_cnt), max_sen_lens):
            with tf.variable_scope('layer' + str(layer)):
                max_doc_len //= max_sen_len
                # x = tf.reshape(x, [-1, max_doc_len, 10, self.hidden_size])
                sen_len = tf.reshape(sen_len, [-1, max_doc_len, max_sen_len])
                sen_len = tf.reduce_sum(1 - tf.cast(tf.equal(sen_len, 0), tf.int32), axis=2)

                x = one_layer(self, x, identity, max_sen_len, sen_len, max_doc_len, convert_flag)

        sen_len = tf.reshape(sen_len, [-1, max_doc_len])
        sen_len = tf.reduce_sum(1 - tf.cast(tf.equal(sen_len, 0), tf.int32), axis=1)
        x = tf.reshape(x, [-1, max_doc_len, self.hidden_size])

        with tf.variable_scope('document_layer'):
            hop_args, attention_args = create_args(identity, sen_len, max_doc_len)
            lstm_outputs, _state = lstm(x, sen_len, self.hidden_size, 'lstm')
            # convert_w = [self.weights['doc_convert_wu'], self.weights['doc_convert_wp']]
            # convert_b = [self.biases['doc_convert_bu'], self.biases['doc_convert_bp']]

            doc_bkg = attention_args['i']
            for ihop in range(self.doc_hop_cnt):
                attention_args['i'] = doc_bkg
                attention_shape = None
                sentence_shape = None
                doc_bkg = hop('hop' + str(ihop), ihop == self.doc_hop_cnt - 1, lstm_outputs,
                              sentence_shape, attention_shape, doc_bkg,
                              hop_args, attention_args, convert_flag)
        outputs = tf.concat(values=doc_bkg, axis=1, name='outputs')

        return outputs

    def build(self, data_iter):
        # get the inputs
        with tf.variable_scope('inputs'):
            input_map = data_iter.get_next()
            usrid, prdid, input_x, input_y, self.doc_len, self.sen_len = \
                (input_map['usr'], input_map['prd'], input_map['content'],
                 input_map['rating'], input_map['doc_len'], input_map['sen_len'])

            usr = lookup(self.embeddings['usr_emb'], usrid, name='cur_usr_embedding')
            prd = lookup(self.embeddings['prd_emb'], prdid, name='cur_prd_embedding')

        # build the process of model
        d_hat, d_hatu, d_hatp = self.dhuapa(input_x, usr, prd, self.convert_flag)
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
