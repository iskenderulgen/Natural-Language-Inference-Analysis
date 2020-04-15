import collections
import importlib
import json
import os

import en_core_web_lg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spacy
import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical

from bert.tokenization import convert_to_unicode

LABELS = {"entailment": 0, "contradiction": 1, "neutral": 2}


def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ["KERAS_BACKEND"] = backend
        importlib.reload(K)
        assert K.backend() == backend
    if backend == "tensorflow":
        K.get_session().close()
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=cfg))
        K.clear_session()


def read_snli(path):
    texts1 = []
    texts2 = []
    labels = []
    with open(path, "r") as file_:
        for line in file_:
            eg = json.loads(line)
            label = eg["gold_label"]
            if label == "-":  # per Parikh, ignore - SNLI entries
                continue
            texts1.append(eg["sentence1"])
            texts2.append(eg["sentence2"])
            labels.append(LABELS[label])
    return texts1, texts2, to_categorical(np.asarray(labels, dtype="int32"))


def load_spacy_nlp(transformer_type):
    nlp = None

    if transformer_type == 'spacy':
        print("Loading spaCy Glove Vectors")
        spacy.prefer_gpu()
        gpu = spacy.require_gpu()
        print("GPU:", gpu)
        nlp = en_core_web_lg.load()

    elif transformer_type == 'fasttext':
        print("Loading fasttext Vectors")
        spacy.prefer_gpu()
        gpu = spacy.require_gpu()
        print("GPU:", gpu)
        nlp = spacy.load("/media/ulgen/Samsung/contradiction_data/Fasttext")

    return nlp


def load_vocab():
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile("/media/ulgen/Samsung/contradiction_data/data/Processed_SNLI/Glove_Processed/vocab.txt",
                        "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def attention_visualization(tokens1, tokens2, attention1, attention2):
    length_sent1 = len(tokens1)
    length_sent2 = len(tokens2)

    attentions_scores = []

    for i in attention1[0][:length_sent1]:
        for j in attention2[0][:length_sent2]:
            attentions_scores.append(np.dot(i, j))
    attentions_scores = np.asarray(attentions_scores)
    attentions_scores = attentions_scores / np.sum(attentions_scores)

    plt.subplots(figsize=(10, 10))

    ax = sns.heatmap(attentions_scores.reshape((length_sent1, length_sent2)), linewidths=0.5, annot=True,
                     cbar=True, cmap="Blues")

    ax.set_yticklabels([i for i in tokens1])
    plt.yticks(rotation=0)
    ax.set_xticklabels([j for j in tokens2])
    plt.show()


def precision(y_true, y_pred):
    y_true, y_pred = K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)
    y_true, y_pred = K.cast(y_true, 'float32'), K.cast(y_pred, 'float32')
    TP = K.sum(K.clip(y_true * y_pred, 0, 1))  # how many
    predicted_positives = K.sum(K.clip(y_pred, 0, 1))
    return TP / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    y_true, y_pred = K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)
    y_true, y_pred = K.cast(y_true, 'float32'), K.cast(y_pred, 'float32')
    TP = K.sum(K.clip(y_true * y_pred, 0, 1))  # how many
    # TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    possible_positives = K.sum(K.clip(y_true, 0, 1))
    return TP / (possible_positives + K.epsilon())


def f1_score(y_true, y_pred):
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    fscore = 2 * (p * r) / (p + r + K.epsilon())
    return fscore
