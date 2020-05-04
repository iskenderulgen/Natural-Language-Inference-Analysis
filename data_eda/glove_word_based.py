import collections
import datetime
import os
import pickle
import random
import tensorflow as tf
import numpy as np

from pretrained_based.utils import read_snli, load_spacy_nlp
from bert.tokenization import convert_to_unicode


def glove_embedding_vocab_extractor(path):
    """Nr_unk for the oov vectors. 100 rows of random vectors"""
    nr_unk = 100

    f = open(os.path.join(path, "Glove/glove.840B.300d.txt"))
    vocab_txt = open(path + "Processed_SNLI/Glove_Processed/vocab.txt", "w")
    vectors = []

    for line in f:
        values = line.split(sep=" ")
        vocab_txt.write(values[0])
        vocab_txt.write("\n")
        vectors.append(np.asarray(values[1:], dtype='float32'))
    f.close()
    vocab_txt.close()

    vectors = np.asarray(vectors)
    oov = np.random.normal(size=(100, 300))
    oov = oov / oov.sum(axis=1, keepdims=True)

    embed_matrix = np.zeros((vectors.shape[0] + nr_unk, 300), dtype="float32")
    embed_matrix[0: nr_unk, ] = oov
    embed_matrix[nr_unk:vectors.shape[0] + nr_unk, ] = vectors

    return embed_matrix


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def create_dataset_ids(nlp, texts, hypotheses, num_unk, max_length, path):
    """This section creates id matrix of the input tokens"""

    sents = texts + hypotheses
    sents_as_ids = []

    print("Total number of sentences to be processed = ", len(sents))
    starttime = datetime.datetime.now()
    count = 0

    vocab = load_vocab(vocab_file=path + "Processed_SNLI/Glove_Processed/vocab.txt")

    for sent in sents:
        doc = nlp(sent, disable=['parser', 'tagger', 'ner', 'textcat'])
        word_ids = []
        for i, token in enumerate(doc):
            if i > max_length:
                break
            if vocab.get(token.text) is not None:
                word_ids.append(num_unk + vocab.get(token.text))
            else:
                word_ids.append(random.randrange(100))

        word_id_vec = np.zeros(max_length, dtype="int")
        clipped_len = min(max_length, len(word_ids))
        word_id_vec[:clipped_len] = word_ids[:clipped_len]
        sents_as_ids.append(word_id_vec)

        count = count + 1
        if count % 50000 == 0:
            print("total sentence: " + str(count) + " Total percent: " + str(count / len(sents)))

    finishtime = datetime.datetime.now()
    totaltime = finishtime - starttime

    print("Total time elapse:" + str(totaltime))
    # text ler ve hipotezleri ayrı ayrı diziler olarak alıyor birinci kısım text - ikinci kısım hipotez
    return [np.array(sents_as_ids[: len(texts)]), np.array(sents_as_ids[len(texts):])]


def glove_word_transformer(path, train_loc, dev_loc, shape, transformer_type):
    print("Transformer type is ", transformer_type)

    nlp = load_spacy_nlp()

    train_texts1, train_texts2, train_labels = read_snli(train_loc)
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)
    print("Processing texts using spacy")

    if os.path.isfile(path=path + "Processed_SNLI/Glove_Processed/vectors.pkl"):
        print("Glove weights matrix already extracted, now loading...")
        with open(path + 'Processed_SNLI/Glove_Processed/vectors.pkl', 'rb') as f:
            vectors = pickle.load(f)
    else:
        print("Glove weight matrix is not found, now extracting...")
        vectors = glove_embedding_vocab_extractor(path=path)
        with open(path + 'Processed_SNLI/Glove_Processed/vectors.pkl', 'wb') as f:
            pickle.dump(vectors, f)

    if os.path.isfile(path=path + "Processed_SNLI/Glove_Processed/train_x.pkl"):
        print("Glove based Pre-Processed train file is found now loading")
        with open(path + 'Processed_SNLI/Glove_Processed/train_x.pkl', 'rb') as f:
            train_x = pickle.load(f)
    else:
        print("There is no Glove based pre-processed file of train_X, Pre-Process will start now")
        train_x = create_dataset_ids(nlp=nlp, texts=train_texts1, hypotheses=train_texts2, num_unk=100,
                                     max_length=shape[0], path=path)
        with open(path + 'Processed_SNLI/Glove_Processed/train_x.pkl', 'wb') as f:
            pickle.dump(train_x, f)

    if os.path.isfile(path=path + "Processed_SNLI/Glove_Processed/dev_x.pkl"):
        print("Glove based Pre-Processed dev file is found now loading")
        with open(path + 'Processed_SNLI/Glove_Processed/dev_x.pkl', 'rb') as f:
            dev_x = pickle.load(f)
    else:
        print("There is no Glove based pre-processed file of dev_X, Pre-Process will start now")
        dev_x = create_dataset_ids(nlp=nlp, texts=dev_texts1, hypotheses=dev_texts2, num_unk=100, max_length=shape[0],
                                   path=path)
        with open(path + 'Processed_SNLI/Glove_Processed/dev_x.pkl', 'wb') as f:
            pickle.dump(dev_x, f)

    return train_x, train_labels, dev_x, dev_labels, vectors
