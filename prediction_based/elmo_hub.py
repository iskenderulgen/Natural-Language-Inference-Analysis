import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from utils.utils import read_snli, load_spacy_nlp


def sentence_transformer(path, premises, hypothesis, nlp):
    sents = premises + hypothesis

    sentence_tokens = []
    tokens_length = []

    for sent in sents:
        tokens = []
        doc = nlp(sent, disable=['parser', 'tagger', 'ner', 'textcat'])
        for token in doc:
            tokens.append(token.text)
        tokens_length.append(len(doc))
        sentence_tokens.append(tokens)

    # max_length = np.amax(tokens_length)

    for sent_token in sentence_tokens:
        while len(sent_token) > 50:
            sent_token.pop()
        while len(sent_token) < 50:
            sent_token.append("")

    for n, i in enumerate(tokens_length):
        if i > 50:
            tokens_length[n] = 50

    elmo = hub.Module(path + "elmo", trainable=True)
    embeddings = elmo(inputs={"tokens": sentence_tokens, "sequence_len": tokens_length},
                      signature="tokens",
                      as_dict=True)["elmo"]
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    embeds = (np.asarray(sess.run(embeddings)))
    # final_weights = np.random.normal(size=(total_senteces, 50, 1024))
    # final_weights = final_weights / final_weights.sum(axis=1, keepdims=True)
    # final_weights[:embeds_list.shape[0], :embeds_list.shape[1], :embeds_list.shape[2]] = embeds_list

    return [(embeds[: len(premises)]), (embeds[len(premises):])]


def elmo_transformer(path, train_loc, dev_loc, feature_type):
    print("Pre - Processing sentences using prediction based ELMO sentence approach")

    train_texts1, train_texts2, train_labels = read_snli(train_loc)
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)
    nlp = load_spacy_nlp(path=path, transformer_type="glove")

    if not os.path.isdir(path + "Processed_SNLI"):
        print("Processed_SNLI directory is not exist, now created")
        os.mkdir(path + "Processed_SNLI")

    if os.path.isfile(path=path + "Processed_SNLI/" + feature_type + "/train_x.pkl"):
        print("Pre-Processed train file is found now loading")
        with open(path + "Processed_SNLI/" + feature_type + "/train_x.pkl", "rb") as f:
            train_x = pickle.load(f)
    else:
        print("There is no pre-processed file of train_X, Pre-Process will start now")
        train_x = sentence_transformer(path=path, premises=train_texts1,
                                       hypothesis=train_texts2, nlp=nlp)
        with open(path + "Processed_SNLI/" + feature_type + "/train_x.pkl", "wb") as f:
            pickle.dump(train_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(path=path + "Processed_SNLI/" + feature_type + "/dev_x.pkl"):
        print("Pre-Processed dev file is found now loading")
        with open(path + "Processed_SNLI/" + feature_type + "/dev_x.pkl", "rb") as f:
            dev_x = pickle.load(f)
    else:
        print("There is no pre-processed file of dev_X, Pre-Process will start now")
        dev_x = sentence_transformer(path=path, premises=dev_texts1,
                                     hypothesis=dev_texts2, nlp=nlp)
        with open(path + "Processed_SNLI/" + feature_type + "/dev_x.pkl", "wb") as f:
            pickle.dump(dev_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_x, train_labels, dev_x, dev_labels
