import tensorflow as tf
from tensorflow import constant as const
from colored import stylize, fg
lookup = tf.nn.embedding_lookup


class DNSC(object):

    def __init__(self, max_sen_len, max_doc_len, cls_cnt, embedding,
                 emb_dim, hidden_size, usr_cnt, prd_cnt, hop_cnt, l2_rate, debug):
        self.max_sen_len = max_sen_len
        self.max_doc_len = max_doc_len
        self.cls_cnt = cls_cnt
        self.embedding = embedding
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.usr_cnt = usr_cnt
        self.prd_cnt = prd_cnt
        self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.hop_cnt = hop_cnt
        self.l2_rate = l2_rate
        self.best_dev_acc = .0
        self.best_test_acc = .0

        # initializers for parameters
        weights_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.contrib.layers.xavier_initializer()
        emb_initializer = tf.initializers.zeros()

        def var(name, shape, initializer):
            return tf.get_variable(name, shape=shape, initializer=initializer)

        # weights in the model
        with tf.name_scope('weights'):
            self.weights = {
                'softmax': var('softmax_w', [hidden_size * 2, cls_cnt], weights_initializer),

                'sen_wh': var('sen_wh', [hidden_size, hidden_size], weights_initializer),
                'sen_wu': var('sen_wu', [hidden_size, hidden_size], weights_initializer),
                'sen_wp': var('sen_wp', [hidden_size, hidden_size], weights_initializer),
                'sen_v': var('sen_v', [hidden_size, 1], weights_initializer),

                'doc_wh': var('doc_wh', [hidden_size, hidden_size], weights_initializer),
                'doc_wu': var('doc_wu', [hidden_size, hidden_size], weights_initializer),
                'doc_wp': var('doc_wp', [hidden_size, hidden_size], weights_initializer),
                'doc_v': var('doc_v', [hidden_size, 1], weights_initializer),

                'sen_convert_wu': var('sen_convert_wu', [hidden_size, hidden_size], weights_initializer),
                'sen_convert_wp': var('sen_convert_wp', [hidden_size, hidden_size], weights_initializer)
            }

        # biases in the model
        with tf.name_scope('biases'):
            self.biases = {
                'softmax': var('softmax_b', [cls_cnt], biases_initializer),

                'sen_attention_b': var('sen_attention_b', [hidden_size], biases_initializer),
                'doc_attention_b': var('doc_attention_b', [hidden_size], biases_initializer)
            }

        # embeddings in the model
        with tf.name_scope('emb'):
            self.embeddings = {
                # 'wrd_emb': const(embedding, name='wrd_emb', dtype=tf.float32),
                'wrd_emb': tf.Variable(embedding, name='wrd_emb', dtype=tf.float32),
                'usr_emb': var('usr_emb', [usr_cnt, hidden_size], emb_initializer),
                'prd_emb': var('prd_emb', [prd_cnt, hidden_size], emb_initializer),
            }

        # for tensorboard
        if debug:
            tf.summary.histogram('usr_emb', self.embeddings['usr_emb'])
            tf.summary.histogram('prd_emb', self.embeddings['prd_emb'])

    def attention(self, v, wh, h, wi, i, b, doc_len, max_len):
        """
        wi, i are two lists where wu, wp and u, p are
        """
        h_shape = h.shape
        # batch_size = h_shape[0]
        max_doc_len = h_shape[1]
        hidden_size = h_shape[2]
        ans = []
        with tf.name_scope('attention'):
            for twi, ti in zip(wi, i):
                h = tf.reshape(h, [-1, hidden_size])
                h = tf.matmul(h, wh)
                e = tf.reshape(h + b, [-1, max_doc_len, hidden_size])
                e = e + tf.matmul(ti, twi)[:, None, :]
                e = tf.tanh(e, name='attention_with_null_word')
                e = tf.reshape(e, [-1, hidden_size])
                e = tf.reshape(tf.matmul(e, v), [-1, max_doc_len])
                e = tf.nn.softmax(e)
                mask = tf.sequence_mask(doc_len, max_doc_len, dtype=tf.float32)
                e = (e * mask)[:, None, :]
                _sum = tf.reduce_sum(e, reduction_indices=2, keepdims=True) + 1e-9
                e = e / _sum
                e = tf.reshape(e, [-1, max_doc_len], name='attention_without_null_word')
                ans.append(e[:, None, :])
        return ans

    def dnsc(self):
        # inputs = tf.reshape(self.x, [-1, self.max_sen_len, self.emb_dim])
        # sen_len = tf.reshape(self.sen_len, [-1])

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

        lstm_outputs, _state = lstm(self.x, self.sen_len, self.hidden_size, 'lstm')
        widentity = [self.weights['sen_wu'], self.weights['sen_wp']]
        identity = [self.usr, self.prd]
        convert_w = [self.weights['sen_convert_wu'], self.weights['sen_convert_wp']]
        new_identity = [0] * len(identity)

        for hop in range(self.hop_cnt):
            with tf.name_scope('hop' + str(hop)):
                sentence = tf.stop_gradient(lstm_outputs) if hop != self.hop_cnt - 1 else lstm_outputs
                # outputs = tf.reshape(outputs, [-1, self.max_doc_len * self.max_sen_len, self.hidden_size])
                alphas = self.attention(self.weights['sen_v'], self.weights['sen_wh'],
                                        sentence, widentity, identity,
                                        self.biases['sen_attention_b'],
                                        self.sen_len, self.max_sen_len)
                # sentence = tf.reshape(sentence, [-1, self.max_sen_len, self.hidden_size])
                for i, alpha in enumerate(alphas):
                    new_identity[i] = tf.matmul(alpha, sentence)
                    # !!! 1. whether to add old embedding
                    # !!! 2. whether to convert old embedding linearly
                    new_identity[i] = tf.reshape(new_identity[i], [-1, self.hidden_size],
                                                 name='new_identity' + str(i))
                    identity[i] = tf.matmul(identity[i], convert_w[i])
                    identity[i] += new_identity[i]
                # outputs = tf.reshape(outputs, [-1, self.max_doc_len, self.hidden_size])

        # with tf.name_scope('doc'):
        #     outputs, state = self.lstm(outputs, self.doc_len, self.hidden_size, 'doc')
        #     beta = self.attention(self.weights['doc_v'], self.weights['doc_wh'],
        #                           outputs, self.weights['doc_wu'], self.usr,
        #                           self.weights['doc_wp'], self.prd, self.biases['doc_attention_b'], self.doc_len, self.max_doc_len)
        #     outputs = tf.matmul(beta, outputs)
        #     d = tf.reshape(outputs, [-1, self.hidden_size])

        # self.alpha = tf.reshape(alpha, [-1, self.max_doc_len, self.max_sen_len])

        with tf.name_scope('result'):
            outputs = tf.concat(values=identity, axis=1, name='hop_outputs')
            d_hat = tf.matmul(outputs, self.weights['softmax']) + self.biases['softmax']
            d_hat = tf.tanh(d_hat, name='d_hat')
        return d_hat

    def build(self, data_iter):
        # get the inputs
        input_map = data_iter.get_next()
        self.usrid, self.prdid, self.input_x, \
            self.input_y, self.sen_len = \
            (input_map['usr'], input_map['prd'], input_map['content'],
             input_map['rating'], input_map['len'])

        self.x = lookup(self.embeddings['wrd_emb'], self.input_x, name='cur_wrd_embedding')
        self.usr = lookup(self.embeddings['usr_emb'], self.usrid, name='cur_usr_embedding')
        self.prd = lookup(self.embeddings['prd_emb'], self.prdid, name='cur_prd_embedding')

        # build the process of model
        self.d_hat = self.dnsc()
        self.prediction = tf.argmax(self.d_hat, 1, name='predictions')

        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.d_hat,
                                                              labels=tf.one_hot(self.input_y, self.cls_cnt))
            regularizer = tf.zeros(1)
            params = tf.trainable_variables()
            for param in params:
                if param not in self.embeddings.values():
                    regularizer += tf.nn.l2_loss(param)
            loss = tf.reduce_sum(loss) + self.l2_rate * regularizer

        with tf.name_scope("metrics"):
            correct_prediction = tf.equal(self.prediction, self.input_y)
            self.mse = tf.reduce_sum(tf.square(self.prediction - self.input_y), name="mse")
            self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.int32), name="correct_num")
            self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"), name="accuracy")

        return loss, self.mse, self.correct_num, self.accuracy

    def output_metrics(self, metrics, data_length):
        loss, mse, correct_num, accuracy = metrics
        info = 'Loss = %.3f, MSE = %.3f, Acc = %.3f' % \
            (loss / data_length, float(mse) / data_length, float(correct_num) / data_length)
        return info

    def record_metrics(self, dev_metrics, test_metrics, devlen, testlen):
        dev_loss, dev_mse, dev_correct_num, dev_accuracy = dev_metrics
        test_loss, test_mse, test_correct_num, test_accuracy = test_metrics
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
