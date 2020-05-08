import importlib
import json
import os

import en_core_web_lg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spacy
from keras import backend as K
from keras.utils import to_categorical

LABELS = {"entailment": 0, "contradiction": 1, "neutral": 2}


def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ["KERAS_BACKEND"] = backend
        importlib.reload(K)
        assert K.backend() == backend
    if backend == "tensorflow":
        K.get_session().close()
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.per_process_memory_fraction = 0.8
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
            if label == "-":  # ignore - SNLI entries
                continue
            texts1.append(eg["sentence1"])
            texts2.append(eg["sentence2"])
            labels.append(LABELS[label])
    return texts1, texts2, to_categorical(np.asarray(labels, dtype="int32"))


def load_spacy_nlp(path, transformer_type):
    nlp = None

    if transformer_type == 'glove':
        print("Loading Glove Vectors")
        spacy.prefer_gpu()
        gpu = spacy.require_gpu()
        print("GPU:", gpu)
        nlp = en_core_web_lg.load()

    elif transformer_type == 'fasttext':
        print("Loading fasttext Vectors")
        spacy.prefer_gpu()
        gpu = spacy.require_gpu()
        print("GPU:", gpu)
        nlp = spacy.load(path + transformer_type)

    elif transformer_type == 'word2vec':
        print("Loading word2vec Vectors")
        spacy.prefer_gpu()
        gpu = spacy.require_gpu()
        print("GPU:", gpu)
        nlp = spacy.load(path + transformer_type)

    return nlp


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
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p * r) / (p + r + K.epsilon())
