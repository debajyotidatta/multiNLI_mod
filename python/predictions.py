"""
Script to generate a CSV file of predictions on the test data.
"""

import tensorflow as tf
import os
import importlib
import random
from util import logger
import util.parameters as params
from util.data_processing_ipynb import *
from util.evaluate import *
import pickle

FIXED_PARAMETERS = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]
logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)

model = FIXED_PARAMETERS["model_type"]

module = importlib.import_module(".".join(['models', model])) 
MyModel = getattr(module, 'MyModel')

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistenyl use the same hyperparameter settings. 
logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)


######################### LOAD DATA #############################

logger.Log("Loading data")
training_snli = load_nli_data(FIXED_PARAMETERS["training_snli"], snli=True)
dev_snli = load_nli_data(FIXED_PARAMETERS["dev_snli"], snli=True)
test_snli = load_nli_data(FIXED_PARAMETERS["test_snli"], snli=True)

training_mnli = load_nli_data(FIXED_PARAMETERS["training_mnli"])
dev_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"])
dev_mismatched = load_nli_data(FIXED_PARAMETERS["dev_mismatched"])
test_matched = load_nli_data(FIXED_PARAMETERS["test_matched"])
test_mismatched = load_nli_data(FIXED_PARAMETERS["test_mismatched"])

dictpath_uni = os.path.join(FIXED_PARAMETERS["log_path"], modname) + "_uni" + ".p"
dictpath_bi = os.path.join(FIXED_PARAMETERS["log_path"], modname) + "_bi" + ".p"
dictpath_tri = os.path.join(FIXED_PARAMETERS["log_path"], modname) + "_tri" + ".p"

if not os.path.isfile(dictpath_uni): 
    print( "No dictionary found!")
    exit(1)

else:
    logger.Log("Loading dictionary from %s" % (dictpath_uni))
    word_indices_uni = pickle.load(open(dictpath_uni, "rb"))
    word_indices_bi = pickle.load(open(dictpath_bi, "rb"))
    word_indices_tri = pickle.load(open(dictpath_tri, "rb"))
    logger.Log("Padding and indexifying sentences")
    sentences_to_padded_index_sequences_ngrams(word_indices_uni, word_indices_bi, word_indices_tri, [training_mnli, training_snli, dev_matched, dev_mismatched, dev_snli, test_snli, test_matched, test_mismatched])

loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices_uni)

logger.Log("Loading embeddings for bigrams")
loaded_embeddings2 = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices_bi)


class modelClassifier:
    def __init__(self, seq_length, loaded_embeddings, loaded_embeddings2):
        ## Define hyperparameters
        self.learning_rate =  FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.emb_train = FIXED_PARAMETERS["emb_train"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"] 
        self.alpha = FIXED_PARAMETERS["alpha"]
        self.word_embeddings = loaded_embeddings

        logger.Log("Building model from %s.py" %(model))
        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=loaded_embeddings, embeddings2=loaded_embeddings2, emb_train=self.emb_train)

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.model.total_cost)

        # tf things: initialize variables and create placeholder for session
        logger.Log("Initializing variables")
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()

    def get_minibatch(self, dataset, start_index, end_index):
        indices = range(start_index, end_index)
        premise_vectors = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence'] for i in indices])
        hypothesis_vectors = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence'] for i in indices])
        premise_vectors_bi = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence_bi'] for i in indices])
        hypothesis_vectors_bi = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence_bi'] for i in indices])
        premise_vectors_tri = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence_tri'] for i in indices])
        hypothesis_vectors_tri = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence_tri'] for i in indices])
        genres = [dataset[i]['genre'] for i in indices]
        labels = [dataset[i]['label'] for i in indices]
        return premise_vectors, hypothesis_vectors, premise_vectors_bi, hypothesis_vectors_bi, premise_vectors_tri, hypothesis_vectors_tri, labels, genres


    def classify(self, examples):
        # This classifies a list of examples
        best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver.restore(self.sess, best_path)
        logger.Log("Model restored from file: %s" % best_path)

        logits = np.empty(3)
        # minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels = self.get_minibatch(examples, 0, len(examples))
        minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_premise_vectors_bi, minibatch_hypothesis_vectors_bi,minibatch_premise_vectors_tri, minibatch_hypothesis_vectors_tri, minibatch_labels, minibatch_genres = self.get_minibatch(examples, 0, len(examples))
     
        
        feed_dict = {self.model.premise_x: minibatch_premise_vectors, 
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.premise_x_bi: minibatch_premise_vectors_bi,
                                self.model.hypothesis_x_bi: minibatch_hypothesis_vectors_bi,
                                self.model.premise_x_tri: minibatch_premise_vectors_tri,
                                self.model.hypothesis_x_tri: minibatch_hypothesis_vectors_tri,
                                self.model.y: minibatch_labels,
                                self.model.keep_rate_ph: 1.0}
        logit = self.sess.run(self.model.logits, feed_dict)
        logits = np.vstack([logits, logit])

        return np.argmax(logits[1:], axis=1)


classifier = modelClassifier(FIXED_PARAMETERS["seq_length"], loaded_embeddings, loaded_embeddings2)

"""
Get CSVs of predictions.
"""

logger.Log("Creating CSV of predicitons on matched test set: %s" %(modname+"_matched_predictions.csv"))
predictions_kaggle(classifier.classify, test_matched, FIXED_PARAMETERS["batch_size"], modname+"_test_matched")

logger.Log("Creating CSV of predicitons on mismatched test set: %s" %(modname+"_mismatched_predictions.csv"))
predictions_kaggle(classifier.classify, test_mismatched, FIXED_PARAMETERS["batch_size"], modname+"_test_mismatched")

