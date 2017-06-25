from __future__ import absolute_import, division
import tensorflow as tf
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils


def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def cnn_encoder(filter_sizes, embedding_size, sequence_length, embedded_chars_expanded, dropout_keep_prob, num_filters):
    embedded_chars_expanded = tf.expand_dims(embedded_chars_expanded, -1)
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        # Convolution Layer
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
            embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = selu(tf.nn.bias_add(conv, b))
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooled_outputs.append(pooled)

        # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    print( "h_drop is",h_drop)
    return num_filters_total, h_drop


class MyModel(object):
    def __init__(self, seq_length, emb_dim, hidden_dim, embeddings, embeddings2, emb_train):
        ## Define hyperparameters
        self.embedding_dim = emb_dim
        # self.dim = hidden_dim
        self.sequence_length = seq_length 

		## Define placeholders
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.premise_x_bi = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.hypothesis_x_bi = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.premise_x_tri = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.hypothesis_x_tri = tf.placeholder(tf.int32, [None, self.sequence_length])
        
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_rate_ph = tf.placeholder(tf.float32, [])

        ## Define remaning parameters 
        # self.E = tf.Variable(embeddings, trainable=emb_train, name="emb")
        with tf.device('/cpu:0'):
            self.E = tf.Variable(tf.random_uniform(embeddings.shape, -1.0,1.0),
                        trainable=True, name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, embeddings.shape)
            self.embedding_init = self.E.assign(self.embedding_placeholder)
            emb_premise = tf.nn.embedding_lookup(self.E, self.premise_x)
            emb_hypothesis = tf.nn.embedding_lookup(self.E, self.hypothesis_x)
        

        with tf.device('/cpu:0'):
            self.E_bi = tf.Variable(tf.random_uniform(embeddings2.shape, -1.0,1.0),
                        trainable=True, name="W")
            emb_premise_bi = tf.nn.embedding_lookup(self.E_bi, self.premise_x_bi)
            emb_hypothesis_bi = tf.nn.embedding_lookup(self.E_bi, self.hypothesis_x_bi) 

        
        def emb_drop_uni(x):
            emb = tf.nn.embedding_lookup(self.E, x)
            emb_drop = tf.nn.dropout(emb, self.keep_rate_ph)
            return emb_drop
        
        def emb_drop_bi(x):
            emb = tf.nn.embedding_lookup(self.E_bi, x)
            emb_drop = tf.nn.dropout(emb, self.keep_rate_ph)
            return emb_drop
        
        premise_in = emb_drop_uni(self.premise_x)
        hypothesis_in = emb_drop_uni(self.hypothesis_x)

        premise_bi = emb_drop_bi(self.premise_x_bi)
        hypothesis_bi = emb_drop_bi(self.hypothesis_x_bi)


        num_filters_total, premise_outs  = cnn_encoder([1,2,3,4,5], self.embedding_dim, self.sequence_length, premise_in, self.keep_rate_ph, num_filters=5)
        _, hypothesis_outs = cnn_encoder([1,2,3,4,5], self.embedding_dim, self.sequence_length, hypothesis_in, self.keep_rate_ph, num_filters=5)


        _, premise_outs_bi  = cnn_encoder([1,2,3,4,5], self.embedding_dim, self.sequence_length, premise_bi, self.keep_rate_ph, num_filters=5)
        _, hypothesis_outs_bi = cnn_encoder([1,2,3,4,5], self.embedding_dim, self.sequence_length, hypothesis_bi, self.keep_rate_ph, num_filters=5)


        final_out = tf.concat([premise_outs, hypothesis_outs, premise_outs_bi, hypothesis_outs_bi], 1)


        self.dim = num_filters_total

        
        self.W0_joined = tf.Variable(tf.random_normal([self.dim * 4, self.dim], stddev=0.1), name="w0")
        self.b0_joined = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b0")

        
        self.W1_joined = tf.Variable(tf.random_normal([self.dim, self.dim], stddev=0.1), name="w1")
        self.b1_joined = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b1")

        self.W2_joined = tf.Variable(tf.random_normal([self.dim, self.dim], stddev=0.1), name="w2")
        self.b2_joined = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b2")

        self.Wcl_joined = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1), name="wcl")
        self.bcl_joined = tf.Variable(tf.random_normal([3], stddev=0.1), name="bcl")
        
        h_1_joined = selu(tf.matmul(final_out, self.W0_joined) + self.b0_joined)
        h_2_joined = selu(tf.matmul(h_1_joined, self.W1_joined) + self.b1_joined)
        h_3_joined = selu(tf.matmul(h_2_joined, self.W2_joined) + self.b2_joined)
        h_drop_joined = tf.nn.dropout(h_3_joined, self.keep_rate_ph)
        

        # Get prediction
        self.logits = tf.matmul(h_drop_joined, self.Wcl_joined) + self.bcl_joined

        # Define the cost function
        self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
