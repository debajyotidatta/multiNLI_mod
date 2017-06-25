import tensorflow as tf
from util import blocks

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
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
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
    def __init__(self, seq_length, emb_dim, hidden_dim, embeddings, emb_train):
        ## Define hyperparameters
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.sequence_length = seq_length 

        ## Define the placeholders
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_rate_ph = tf.placeholder(tf.float32, [])

        ## Define parameters
        self.E = tf.Variable(embeddings, trainable=emb_train)
        
        # self.W_mlp = tf.Variable(tf.random_normal([self.dim * 8, self.dim], stddev=0.1))
        # self.b_mlp = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

        # self.W_cl = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1))
        # self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1))
        
        ## Function for embedding lookup and dropout at embedding layer
        def emb_drop(x):
            emb = tf.nn.embedding_lookup(self.E, x)
            emb_drop = tf.nn.dropout(emb, self.keep_rate_ph)
            return emb_drop

        # Get lengths of unpadded sentences
        prem_seq_lengths, prem_mask = blocks.length(self.premise_x)
        hyp_seq_lengths, hyp_mask = blocks.length(self.hypothesis_x)


        ### BiLSTM layer ###
        premise_in = emb_drop(self.premise_x)
        hypothesis_in = emb_drop(self.hypothesis_x)

        num_filters_total, premise_outs  = cnn_encoder([1,2,3,4,5], self.embedding_dim, self.sequence_length, premise_in, self.keep_rate_ph, num_filters=5)
        num_filters_total, hypothesis_outs = cnn_encoder([1,2,3,4,5], self.embedding_dim, self.sequence_length, hypothesis_in, self.keep_rate_ph, num_filters=5)
        print ("from cnn", premise_outs, hypothesis_outs)
        # premise_outs, c1 = blocks.biLSTM(premise_in, dim=self.dim, seq_len=prem_seq_lengths, name='premise')
        # hypothesis_outs, c2 = blocks.biLSTM(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths, name='hypothesis')
        # print ("from lstms", premise_outs, hypothesis_outs)
        final_out = tf.concat([premise_outs, hypothesis_outs], 1)
        print("final_out", final_out)

        W = tf.get_variable(
                "W",
                shape=[num_filters_total*2, 3],
                initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[3]), name="b")

            # scores = tf.nn.xw_plus_b(final_out, W, b, name="scores")
            # predictions = tf.argmax(scores, 1, name="predictions")



        # # # MLP layer
        # h_mlp = tf.nn.relu(tf.matmul(final_out, W) + b)
        # # # Dropout applied to classifier
        # h_drop = tf.nn.dropout(h_mlp, self.keep_rate_ph)

        # # Get prediction
        self.logits = tf.matmul(final_out, W) + b
        print("logits", self.logits)

        # # Define the cost function
        self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
