"""
This code converts premises and hypothesis using pre trained word weights. Currently it supports 3
(word2vec, glove, fasttext) word weights. All are pruned to 685k unique vectors. Pruning conducted
based on spacy's init module. Unique vector size referred from original spacy's glove weight size.
"""

import datetime
import os
import pickle
import cupy as cp
import numpy as np

from pretrained_based.utils import read_snli, load_spacy_nlp


def create_dataset_ids(nlp, premises, hypothesis, num_unk, max_length):
    """This section creates id matrix of the input tokens"""

    sents = premises + hypothesis
    sents_as_ids = []

    print("Total number of sentences to be processed = ", len(sents))
    start_time = datetime.datetime.now()
    sent_count = 0

    for sent in sents:
        doc = nlp(sent, disable=['parser', 'tagger', 'ner', 'textcat'])
        word_ids = []
        for i, token in enumerate(doc):
            if token.has_vector and token.vector_norm == 0:
                continue
            if i > max_length:
                break
            if token.has_vector:
                word_ids.append(token.rank + num_unk + 1)
            else:
                # if we don't have a vector, pick an OOV entry
                word_ids.append(token.rank % num_unk + 1)

        # there must be a simpler way of generating padded arrays from lists...
        word_id_vec = np.zeros(max_length, dtype="int")
        clipped_len = min(max_length, len(word_ids))
        word_id_vec[:clipped_len] = word_ids[:clipped_len]
        sents_as_ids.append(word_id_vec)

        sent_count = sent_count + 1
        if sent_count % 50000 == 0:
            print("processed Sentence: " + str(sent_count) + " Processed Percentage: " +
                  str(round(sent_count / len(sents), 4) * 100))

    finish_time = datetime.datetime.now()
    total_time = finish_time - start_time
    print("Total time spent to create token ID's of sentences: ", total_time)

    return [np.array(sents_as_ids[: len(premises)]), np.array(sents_as_ids[len(premises):])]


def get_embeddings(vocab, nr_unk=100):
    # the extra +1 is for a zero vector representing sentence-final padding
    num_vectors = max(lex.rank for lex in vocab) + 2

    # create random vectors for OOV tokens
    oov = np.random.normal(size=(nr_unk, vocab.vectors_length))
    oov = oov / oov.sum(axis=1, keepdims=True)

    vectors = np.zeros((num_vectors + nr_unk, vocab.vectors_length), dtype="float32")
    vectors[1: (nr_unk + 1), ] = oov
    for lex in vocab:
        if lex.has_vector and lex.vector_norm > 0:
            vectors[nr_unk + lex.rank + 1] = cp.asnumpy(lex.vector / lex.vector_norm)

    print("extracting embeddings is finished")

    return vectors


def spacy_word_transformer(path, train_loc, dev_loc, shape, transformer_type):
    print("Starting to process using spacy. Transformer type is ", transformer_type)
    nlp = load_spacy_nlp(path=path, transformer_type=transformer_type)

    train_texts1, train_texts2, train_labels = read_snli(train_loc)
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)

    if os.path.isdir(path + "Processed_SNLI"):
        print("Processed_SNLI directory is exist")
    else:
        os.mkdir(path + "Processed_SNLI")
        print("Processed_SNLI directory is created")

    if os.path.isfile(path=path + "Processed_SNLI/" + transformer_type + "/train_x.pkl"):
        print(transformer_type, "based Pre-Processed train file is found now loading...")
        with open(path + "Processed_SNLI/" + transformer_type + "/train_x.pkl", "rb") as f:
            train_x = pickle.load(f)
    else:
        print(transformer_type, "based pre-processed file of train_X isn't exist, Pre-Process will start now")
        train_x = create_dataset_ids(nlp=nlp, premises=train_texts1, hypothesis=train_texts2, num_unk=100,
                                     max_length=shape[0])
        with open(path + "Processed_SNLI/" + transformer_type + "/train_x.pkl", "wb") as f:
            pickle.dump(train_x, f)

    if os.path.isfile(path=path + "Processed_SNLI/" + transformer_type + "/dev_x.pkl"):
        print(transformer_type, "based Pre-Processed dev file is found now loading...")
        with open(path + "Processed_SNLI/" + transformer_type + "/dev_x.pkl", "rb") as f:
            dev_x = pickle.load(f)
    else:
        print(transformer_type, "based pre-processed file of dev_X isn't exist, Pre-Process will start now")
        dev_x = create_dataset_ids(nlp=nlp, premises=dev_texts1, hypothesis=dev_texts2, num_unk=100,
                                   max_length=shape[0])
        with open(path + "Processed_SNLI/" + transformer_type + "/dev_x.pkl", "wb") as f:
            pickle.dump(dev_x, f)

    if os.path.isfile(path=path + "Processed_SNLI/" + transformer_type + "/weights.pkl"):
        print(transformer_type, "weights matrix already extracted, now loading...")
        with open(path + "Processed_SNLI/" + transformer_type + "/spacy_weights.pkl", "rb") as f:
            vectors = pickle.load(f)
    else:
        print(transformer_type, " weight matrix is not found, now extracting...")
        vectors = get_embeddings(vocab=nlp.vocab, nr_unk=100)
        with open(path + "Processed_SNLI/" + transformer_type + "/spacy_weights.pkl", "wb") as f:
            pickle.dump(vectors, f)

    return train_x, train_labels, dev_x, dev_labels, vectors
