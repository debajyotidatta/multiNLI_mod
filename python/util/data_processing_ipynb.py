import numpy as np
import re
import random
import json
import collections
import parameters as params
import pickle
import nltk

# args = params.argparser("lstm petModel-0 --keep_rate 0.9 --seq_length 25 --emb_train")

# FIXED_PARAMETERS = params.load_parameters(args)

FIXED_PARAMETERS = params.load_parameters()

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0
}

PADDING = "<PAD>"
UNKNOWN = "<UNK>"

def load_nli_data(path, snli=False):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    data = []
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data

def load_nli_data_genre(path, genre, snli=True):
    """
    Load a specific genre's examples from MultiNLI, or load SNLI data and assign a "snli" genre to the examples.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. If set to true, it will overwrite the genre label for MultiNLI data.
    """
    data = []
    j = 0
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            if loaded_example["genre"] == genre:
                data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data

def tokenize(string):
    string = re.sub(r'\(|\)', '', string)
    return string.split()

def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """  
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['sentence1_binary_parse']))
            word_counter.update(tokenize(example['sentence2_binary_parse']))
        
    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary
        
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices


def build_dictionary_ngrams(training_datasets):
    """
    Extract vocabulary and build bi and trigram dictionaries.
    """  
    word_counter_unigrams = collections.Counter()
    word_counter_bigrams = collections.Counter()
    word_counter_trigrams = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            sent1_tokenized = tokenize(example['sentence1_binary_parse'])
            sent2_tokenized = tokenize(example['sentence2_binary_parse'])
            bigrams1 = nltk.bigrams(sent1_tokenized)
            bigrams2 = nltk.bigrams(sent2_tokenized)
            trigrams1 = nltk.trigrams(sent1_tokenized)
            trigrams2 = nltk.trigrams(sent2_tokenized)
            word_counter_bigrams.update(bigrams1)
            word_counter_bigrams.update(bigrams2)
            word_counter_trigrams.update(trigrams1)
            word_counter_trigrams.update(trigrams2)
            word_counter_unigrams.update(sent1_tokenized)
            word_counter_unigrams.update(sent2_tokenized)
        
    vocabulary_uni = set([word for word in word_counter_unigrams])
    vocabulary_uni = list(vocabulary_uni)
    vocabulary_uni = [PADDING, UNKNOWN] + vocabulary_uni   
    word_indices_uni = dict(zip(vocabulary_uni, range(len(vocabulary_uni))))
    
    vocabulary_bi = set([word for word in word_counter_bigrams])
    vocabulary_bi = list(vocabulary_bi)
    vocabulary_bi = [PADDING, UNKNOWN] + vocabulary_bi    
    word_indices_bi = dict(zip(vocabulary_bi, range(len(vocabulary_bi))))
    
    vocabulary_tri = set([word for word in word_counter_trigrams])
    vocabulary_tri = list(vocabulary_tri)
    vocabulary_tri = [PADDING, UNKNOWN] + vocabulary_tri  
    word_indices_tri = dict(zip(vocabulary_tri, range(len(vocabulary_tri))))

    return word_indices_uni, word_indices_bi, word_indices_tri



def sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    for i, dataset in enumerate(datasets):
        for example in dataset:
            for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
                # print("sentence is", sentence)
                example[sentence + '_index_sequence'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)

                token_sequence = tokenize(example[sentence])
                padding = FIXED_PARAMETERS["seq_length"] - len(token_sequence)

                for i in range(FIXED_PARAMETERS["seq_length"]):
                    if i >= len(token_sequence):
                        index = word_indices[PADDING]
                    else:
                        if token_sequence[i] in word_indices:
                            index = word_indices[token_sequence[i]]
                        else:
                            index = word_indices[UNKNOWN]
                    example[sentence + '_index_sequence'][i] = index

                    

def sentences_to_padded_index_sequences_ngrams(word_indices, word_indices_bi, word_indices_tri, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    for i, dataset in enumerate(datasets):
        for example in dataset:
            for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
                # print("sentence is", sentence)
                example[sentence + '_index_sequence'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)
                example[sentence + '_index_sequence_bi'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)
                example[sentence + '_index_sequence_tri'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)

                token_sequence = tokenize(example[sentence])
                padding = FIXED_PARAMETERS["seq_length"] - len(token_sequence)

                for i in range(FIXED_PARAMETERS["seq_length"]):
                    if i >= len(token_sequence):
                        index = word_indices[PADDING]
                    else:
                        if token_sequence[i] in word_indices:
                            index = word_indices[token_sequence[i]]
                        else:
                            index = word_indices[UNKNOWN]
                    example[sentence + '_index_sequence'][i] = index
                
                token_sequence_bi = list(nltk.bigrams(token_sequence))
                padding_bi = FIXED_PARAMETERS["seq_length"] - len(token_sequence_bi)

                for i in range(FIXED_PARAMETERS["seq_length"]):
                    if i >= len(token_sequence_bi):
                        index = word_indices_bi[PADDING]
                    else:
                        if token_sequence_bi[i] in word_indices_bi:
                            index = word_indices_bi[token_sequence_bi[i]]
                        else:
                            index = word_indices_bi[UNKNOWN]
                    example[sentence + '_index_sequence_bi'][i] = index
                    
                    
                token_sequence_tri = list(nltk.trigrams(token_sequence))
                padding_tri = FIXED_PARAMETERS["seq_length"] - len(token_sequence_tri)

                for i in range(FIXED_PARAMETERS["seq_length"]):
                    if i >= len(token_sequence_tri):
                        index = word_indices_tri[PADDING]
                    else:
                        if token_sequence_tri[i] in word_indices_tri:
                            index = word_indices_tri[token_sequence_tri[i]]
                        else:
                            index = word_indices_tri[UNKNOWN]
                    example[sentence + '_index_sequence_tri'][i] = index
                    

def loadEmbedding_zeros(path, word_indices):
    """
    Load GloVe embeddings. Initializng OOV words to vector of zeros.
    """
    emb = np.zeros((len(word_indices), FIXED_PARAMETERS["word_embedding_dim"]), dtype='float32')
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if FIXED_PARAMETERS["embeddings_to_load"] != None:
                if i >= FIXED_PARAMETERS["embeddings_to_load"]:
                    break
            
            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb


def loadEmbedding_rand(path, word_indices):
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    n = len(word_indices)
    m = FIXED_PARAMETERS["word_embedding_dim"]
    emb = np.empty((n, m), dtype=np.float32)

    emb[:,:] = np.random.normal(size=(n,m))

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0:2, :] = np.zeros((1,m), dtype="float32")
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if FIXED_PARAMETERS["embeddings_to_load"] != None:
                if i >= FIXED_PARAMETERS["embeddings_to_load"]:
                    break
            
            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb

