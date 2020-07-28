import tensorflow as tf
import numpy as np

def lrelu(x, n="lrelu", leak=0.2): 
    return tf.maximum(x, leak * x, name=n) 

f_h_layer_size = 300
s_h_layer_size = 100

class NLPCNN(object):
    
    def __init__(self, max_seq_len, num_classes, embedding_size, filter_sizes, num_filters, l2_lambda=0.05, learning_rate=0.001, train_on_gpu=True, max_grad_norm=2, noise=False):
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.total_filter_count = self.num_filters * len(filter_sizes)
        self.conv_w_summaries = []
        self.conv_b_summaries = []

        self.input_x = tf.placeholder(tf.float32, [None, max_seq_len, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.int8, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep_prob")
        self.tf_accuracy_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_ph')
        self.tf_test_loss_ph = tf.placeholder(tf.float32,shape=None, name='test_loss_ph')
        self.tf_train_accuracy_ph = tf.placeholder(tf.float32,shape=None, name='train_accuracy_ph')
        self.is_training = tf.placeholder_with_default(False, shape=(), name="is_training")
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.l2_loss = tf.constant(0.0)

        if noise == True:
            self.input_x += tf.cond(self.is_training, lambda: tf.random_normal(shape=tf.shape(self.input_x), mean=0.0, stddev=16), lambda: tf.zeros(shape=tf.shape(self.input_x)))

        self.__create_all_convolutions()

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            initializer = tf.contrib.layers.xavier_initializer()
            W = tf.Variable(initializer([self.total_filter_count, f_h_layer_size]), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[f_h_layer_size]), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b)
            self.logits = tf.nn.relu(self.logits)

            W2 = tf.Variable(initializer([f_h_layer_size, s_h_layer_size]), name="W")
            b2 = tf.Variable(tf.constant(0.1, shape=[s_h_layer_size]), name="b")
            self.logits = tf.nn.dropout(self.logits, self.dropout_keep_prob)
            self.logits = tf.nn.xw_plus_b(self.logits, W2, b2, name="logits")
            self.logits = tf.nn.relu(self.logits)

            W3 = tf.Variable(initializer([s_h_layer_size, self.num_classes]), name="W")
            b3 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.logits = tf.nn.dropout(self.logits, self.dropout_keep_prob)
            self.logits = tf.nn.xw_plus_b(self.logits, W3, b3)

            self.prediction = tf.nn.softmax(self.logits, name="prediction")
            self.prediction_argmax = tf.argmax(self.prediction, axis=1, name="prediction_argmax")
            self.output_w_summary = tf.summary.histogram("weights", W)
            self.output_b_summary = tf.summary.histogram("biases", b)

        with tf.name_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.input_y, tf.float32))
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.loss = tf.reduce_mean(self.loss) + l2_lambda * self.l2_loss
            self.loss_summary = tf.summary.scalar("train_loss", self.loss)
            self.test_loss_summary = tf.summary.scalar("test_loss", self.tf_test_loss_ph)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(self.prediction_argmax, tf.argmax(self.input_y, axis=1))
            self.correct_count = tf.count_nonzero(tf.cast(correct_pred, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            self.acc_summary = tf.summary.scalar("test_accuracy", self.tf_accuracy_ph)
            self.train_acc_summary = tf.summary.scalar("train_accuracy", self.tf_train_accuracy_ph)
        
        with tf.name_scope("optimization"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            tvars = tf.trainable_variables()
            gradients = tf.gradients(self.loss, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
            
            device = "/gpu:0" if train_on_gpu else "/cpu:0"

            with tf.device(device):
                grads, global_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


    def __create_convolution(self, filter_size):     
        with tf.name_scope("conv-%s" % filter_size):
            filter_shape = [filter_size, self.embedding_size, self.num_filters]
            
            initializer = tf.contrib.layers.xavier_initializer()
            W = tf.Variable(initializer(filter_shape), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
            self.conv_w_summaries.append(tf.summary.histogram("weights", W))
            self.conv_b_summaries.append(tf.summary.histogram("biases", b))
            
            conv = tf.nn.conv1d(
                self.input_x,
                W,
                stride=1,
                padding="VALID",
                name="conv"
            )
            
            h = lrelu(tf.nn.bias_add(conv, b), "lrelu")
            
            pool = tf.layers.max_pooling1d(
                h,
                pool_size=self.max_seq_len - filter_size + 1,
                strides=1,
                padding="VALID",
                name="pool"
            )

            return pool

    def __create_all_convolutions(self):
        pooled_outputs = []

        for i, filter_size in enumerate(self.filter_sizes):
            pool = self.__create_convolution(filter_size)
            pooled_outputs.append(pool)

        self.h_pool = tf.concat(pooled_outputs, 2)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.total_filter_count])
