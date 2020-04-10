import collections
import json

import en_core_web_lg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spacy
import tensorflow as tf
from keras.utils import to_categorical

from bert.tokenization import convert_to_unicode

LABELS = {"entailment": 0, "contradiction": 1, "neutral": 2}


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


def load_spacy_nlp():
    print("Loading spaCy")
    spacy.prefer_gpu()
    gpu = spacy.require_gpu()
    print("GPU:", gpu)
    nlp = en_core_web_lg.load()

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
