import json

import en_core_web_lg
import numpy as np
import spacy
from keras.utils import to_categorical

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
