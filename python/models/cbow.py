import tensorflow as tf

class MyModel(object):
    def __init__(self, seq_length, emb_dim, hidden_dim, embeddings, embeddings2, emb_train):
        ## Define hyperparameters
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
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
                
                
                # self.E = tf.Variable(tf.random_uniform(embeddings.shape, -1.0,1.0),
                #         trainable=True, name="W")





        

        self.W_0 = tf.Variable(tf.random_normal([self.embedding_dim * 2, self.dim], stddev=0.1), name="w0")
        self.b_0 = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b0")

        self.W_1 = tf.Variable(tf.random_normal([self.dim, self.dim], stddev=0.1), name="w1")
        self.b_1 = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b1")

        self.W_2 = tf.Variable(tf.random_normal([self.dim, self.dim], stddev=0.1), name="w2")
        self.b_2 = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b2")

        
        
        ## Calculate representaitons by CBOW method
         
        emb_premise_drop = tf.nn.dropout(emb_premise, self.keep_rate_ph)

        
        emb_hypothesis_drop = tf.nn.dropout(emb_hypothesis, self.keep_rate_ph)

        premise_rep = tf.reduce_sum(emb_premise_drop, 1)
        hypothesis_rep = tf.reduce_sum(emb_hypothesis_drop, 1)

        ## Combinations
        # h_diff = premise_rep - hypothesis_rep
        # h_mul = premise_rep * hypothesis_rep
        
        
        
        # self.E_bi = tf.Variable(embeddings2, name="emb2")
        with tf.device('/cpu:0'):
                self.E_bi = tf.Variable(tf.random_uniform(embeddings2.shape, -1.0,1.0),
                        trainable=True, name="W")
                # self.embedding_placeholder_bi = tf.placeholder(tf.float32, embeddings2.shape)
                # self.embedding2_init = self.E_bi.assign(self.embedding_placeholder_bi)
                emb_premise_bi = tf.nn.embedding_lookup(self.E_bi, self.premise_x_bi)
                emb_hypothesis_bi = tf.nn.embedding_lookup(self.E_bi, self.hypothesis_x_bi) 




        self.W_0_bi = tf.Variable(tf.random_normal([self.embedding_dim * 2, self.dim], stddev=0.1), name="w0")
        self.b_0_bi = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b0")

        self.W_1_bi = tf.Variable(tf.random_normal([self.dim, self.dim], stddev=0.1), name="w1")
        self.b_1_bi = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b1")

        self.W_2_bi = tf.Variable(tf.random_normal([self.dim, self.dim], stddev=0.1), name="w2")
        self.b_2_bi = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b2")
        
        
        ## Calculate representaitons by CBOW method
        
        emb_premise_drop_bi = tf.nn.dropout(emb_premise_bi, self.keep_rate_ph)

        
        emb_hypothesis_drop_bi = tf.nn.dropout(emb_hypothesis_bi, self.keep_rate_ph)

        premise_rep_bi = tf.reduce_sum(emb_premise_drop_bi, 1)
        hypothesis_rep_bi = tf.reduce_sum(emb_hypothesis_drop_bi, 1)

        ## Combinations
        # h_diff_bi = premise_rep_bi - hypothesis_rep_bi
        # h_mul_bi = premise_rep_bi * hypothesis_rep_bi
        
        
        ### MLP for unigrams

        # mlp_input = tf.concat([premise_rep, hypothesis_rep, h_diff, h_mul], 1)
        mlp_input = tf.concat([premise_rep, hypothesis_rep], 1)
        
        h_1 = tf.nn.relu(tf.matmul(mlp_input, self.W_0) + self.b_0)
        h_2 = tf.nn.relu(tf.matmul(h_1, self.W_1) + self.b_1)
        h_3 = tf.nn.relu(tf.matmul(h_2, self.W_2) + self.b_2)
        h_drop = tf.nn.dropout(h_3, self.keep_rate_ph)
        
        
        ### MLP for bigrams
        
        # mlp_input_bi = tf.concat([premise_rep_bi, hypothesis_rep_bi, h_diff_bi, h_mul_bi], 1)
        mlp_input_bi = tf.concat([premise_rep_bi, hypothesis_rep_bi], 1)
        
        h_1_bi = tf.nn.relu(tf.matmul(mlp_input_bi, self.W_0_bi) + self.b_0_bi)
        h_2_bi = tf.nn.relu(tf.matmul(h_1_bi, self.W_1_bi) + self.b_1_bi)
        h_3_bi = tf.nn.relu(tf.matmul(h_2_bi, self.W_2_bi) + self.b_2_bi)
        h_drop_bi = tf.nn.dropout(h_3_bi, self.keep_rate_ph)
        
        
        ###
        
        uni_out = tf.nn.relu(h_drop)
        bi_out = tf.nn.relu(h_drop_bi)
        
        self.W0_joined = tf.Variable(tf.random_normal([self.dim * 2, self.dim], stddev=0.1), name="w0")
        self.b0_joined = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b0")

        
        self.W1_joined = tf.Variable(tf.random_normal([self.dim, self.dim], stddev=0.1), name="w1")
        self.b1_joined = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b1")

        self.W2_joined = tf.Variable(tf.random_normal([self.dim, self.dim], stddev=0.1), name="w2")
        self.b2_joined = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b2")

        self.Wcl_joined = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1), name="wcl")
        self.bcl_joined = tf.Variable(tf.random_normal([3], stddev=0.1), name="bcl")
        
        
        
        fin_output = tf.concat([uni_out, bi_out], 1)
        
        h_1_joined = tf.nn.relu(tf.matmul(fin_output, self.W0_joined) + self.b0_joined)
        h_2_joined = tf.nn.relu(tf.matmul(h_1_joined, self.W1_joined) + self.b1_joined)
        h_3_joined = tf.nn.relu(tf.matmul(h_2_joined, self.W2_joined) + self.b2_joined)
        h_drop_joined = tf.nn.dropout(h_3_joined, self.keep_rate_ph)
        

        # Get prediction
        self.logits = tf.matmul(h_drop_joined, self.Wcl_joined) + self.bcl_joined

        # Define the cost function
        self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
