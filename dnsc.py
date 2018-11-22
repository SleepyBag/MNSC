import tensorflow as tf
from tensorflow import get_variable as var
from tensorflow import constant as const


class DNSC(object):

    def __init__(self, max_sen_len, max_doc_len, cls_cnt, emb_file,
                 emb_dim, hidden_size, usr_cnt, prd_cnt, hop_cnt, l2_rate):
        self.max_sen_len = max_sen_len
        self.max_doc_len = max_doc_len
        self.cls_cnt = cls_cnt
        self.emb_file = emb_file
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.usr_cnt = usr_cnt
        self.prd_cnt = prd_cnt
        self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.hop_cnt = hop_cnt
        self.l2_rate = l2_rate

        with tf.name_scope('inputs'):
            self.usrid = tf.placeholder(tf.int32, [None], name="usrid")
            self.prdid = tf.placeholder(tf.int32, [None], name="prdid")
            self.input_x = tf.placeholder(tf.int32, [None, self.max_doc_len, self.max_sen_len], name="input_x")
            self.input_y = tf.placeholder(tf.int32, [None, self.cls_cnt], name="input_y")
            self.sen_len = tf.placeholder(tf.int32, [None, self.max_doc_len], name="sen_len")
            self.doc_len = tf.placeholder(tf.int32, [None], name="doc_len")

        with tf.name_scope('weights'):
            self.weights = {
                'softmax': tf.get_variable('softmax_w', shape=[self.hidden_size * 2, self.cls_cnt], initializer=tf.contrib.layers.xavier_initializer()),

                'sen_wh': tf.get_variable('sen_wh', shape=[self.hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
                'sen_wu': tf.get_variable('sen_wu', shape=[self.hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
                'sen_wp': tf.get_variable('sen_wp', shape=[self.hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
                'sen_v': tf.get_variable('sen_v', shape=[self.hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer()),

                'doc_wh': tf.get_variable('doc_wh', shape=[self.hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
                'doc_wu': tf.get_variable('doc_wu', shape=[self.hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
                'doc_wp': tf.get_variable('doc_wp', shape=[self.hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
                'doc_v': tf.get_variable('doc_v', shape=[self.hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax': tf.get_variable('softmax_b', shape=[self.cls_cnt], initializer=tf.contrib.layers.xavier_initializer()),

                'sen_attention_b': tf.get_variable('sen_attention_b', shape=[self.hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
                'doc_attention_b': tf.get_variable('doc_attention_b', shape=[self.hidden_size], initializer=tf.contrib.layers.xavier_initializer())
            }

        with tf.name_scope('emb'):
            self.embeddings = {
                # 'wrd_emb': tf.constant(self.emb_file, name='wrd_emb', dtype=tf.float32),
                'wrd_emb': tf.Variable(self.emb_file, name='wrd_emb', dtype=tf.float32),
                'usr_emb': tf.get_variable('usr_emb', shape=[self.usr_cnt, self.hidden_size], initializer=tf.initializers.zeros()),
                'prd_emb': tf.get_variable('prd_emb', shape=[self.prd_cnt, self.hidden_size], initializer=tf.initializers.zeros()),
            }
        self.x = tf.nn.embedding_lookup(self.embeddings['wrd_emb'], self.input_x)
        self.usr = tf.nn.embedding_lookup(self.embeddings['usr_emb'], self.usrid)
        self.prd = tf.nn.embedding_lookup(self.embeddings['prd_emb'], self.prdid)

    def lstm(self, inputs, sequence_length, hidden_size, scope):
        outputs, state = tf.nn.dynamic_rnn(
            cell=tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=0., initializer=tf.contrib.layers.xavier_initializer()),
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32,
            scope=scope
        )
        return outputs, state

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
                # u = tf.matmul(u, wu)
                # p = tf.matmul(p, wp)
                e = e + tf.matmul(ti, twi)[:, None, :]
                # e = e + u[:, None, :] + p[:, None, :]
                e = tf.tanh(e)
                e = tf.reshape(e, [-1, hidden_size])
                e = tf.reshape(tf.matmul(e, v), [-1, max_len])
                e = tf.nn.softmax(e)
                mask = tf.sequence_mask(doc_len, max_len, dtype=tf.float32)
                e = (e * mask)[:, None, :]
                _sum = tf.reduce_sum(e, reduction_indices=2, keepdims=True) + 1e-9
                e = e / _sum
                e = tf.reshape(e, [-1, max_len])
        return ans

    def dnsc(self):
        inputs = tf.reshape(self.x, [-1, self.max_sen_len, self.emb_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        lstm_outputs, _state = self.lstm(inputs, sen_len, self.hidden_size, 'lstm')
        widentity = [self.weights['sen_wu'], self.weights['sen_wp']]
        identity = [self.usr, self.prd]

        for hop in range(self.hop_cnt):
            with tf.name_scope('hop' + str(hop)):
                sentence = tf.stop_gradient(lstm_outputs) if hop != self.hop_cnt - 1 else lstm_outputs
                # outputs = tf.reshape(outputs, [-1, self.max_doc_len * self.max_sen_len, self.hidden_size])
                alphas = self.attention(self.weights['sen_v'], self.weights['sen_wh'],
                                        sentence, widentity, identity,
                                        self.biases['sen_attention_b'],
                                        sen_len, self.max_sen_len)
                # sentence = tf.reshape(sentence, [-1, self.max_sen_len, self.hidden_size])
                for i, alpha in enumerate(alphas):
                    identity[i] = tf.matmul(alpha, sentence)
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
            outputs = tf.concat(values=identity, axis=1)
            d_hat = tf.tanh(tf.matmul(outputs, self.weights['softmax']) + self.biases['softmax'])
            # p_hat = tf.nn.softmax(d_hat)
        return d_hat

    def build(self):
        self.d_hat = self.dnsc()
        self.prediction = tf.argmax(self.d_hat, 1, name='predictions')

        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.d_hat, labels=self.input_y)
            regularizer = tf.zeros(1)
            params = tf.trainable_variables()
            for param in params:
                if param not in self.embeddings.values():
                    regularizer += tf.nn.l2_loss(param)
            loss = tf.reduce_mean(loss) + self.l2_rate * regularizer

        with tf.name_scope("metrics"):
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.mse = tf.reduce_sum(tf.square(self.prediction - tf.argmax(self.input_y, 1)), name="mse")
            self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.int32), name="correct_num")
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

        return loss, self.mse, self.correct_num, self.accuracy
