import tensorflow as tf
from util import blocks
from util import selu as selu

# implementation from https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py

def attention(inputs, attention_size, time_major=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    inputs_shape = inputs.shape
    sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    return output


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
        h = selu.selu(tf.nn.bias_add(conv, b), name="selu")
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
        

        premise_outs_attention = attention(premise_outs, attention_size=num_filters_total)
        hypothesis_outs_attention = attention(hypothesis_outs, attention_size=num_filters_total)
        
        # print ("from cnn", premise_outs, hypothesis_outs)
        # premise_outs, c1 = blocks.biLSTM(premise_in, dim=self.dim, seq_len=prem_seq_lengths, name='premise')
        # hypothesis_outs, c2 = blocks.biLSTM(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths, name='hypothesis')
        # print ("from lstms", premise_outs, hypothesis_outs)
        final_out = tf.concat([premise_outs_attention, hypothesis_outs_attention], 1)
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
