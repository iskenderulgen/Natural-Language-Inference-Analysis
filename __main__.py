import datetime
import numpy as np
import json
import plac
import sys
import os
import importlib
import spacy
import en_core_web_lg
import os.path
import tensorflow as tf

from bert import extract_features
from keras.utils import to_categorical
from keras_decomposable_attention import build_model, build_model_bert
from spacy_hook import get_embeddings, KerasSimilarityShim
from keras import backend as k_backend

import pickle

path = "/home/ulgen/Documents/Python_Projects/Contradiction/data/"


# workaround for keras/tensorflow bug
# see https://github.com/tensorflow/tensorflow/issues/3388


def set_keras_backend(backend):
    if k_backend.backend() != backend:
        os.environ["KERAS_BACKEND"] = backend
        importlib.reload(k_backend)
        assert k_backend.backend() == backend
    if backend == "tensorflow":
        k_backend.get_session().close()
        cfg = k_backend.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        k_backend.set_session(k_backend.tf.Session(config=cfg))
        k_backend.clear_session()


set_keras_backend("tensorflow")
LABELS = {"entailment": 0, "contradiction": 1, "neutral": 2}


def train(train_loc, dev_loc, shape, settings, bert_path, transformer_type):
    print("Loading spaCy")
    spacy.prefer_gpu()
    gpu = spacy.require_gpu()
    print("GPU:", gpu)
    nlp = en_core_web_lg.load()

    assert nlp.path is not None
    print("Transformer type is = " + transformer_type)
    train_x, train_labels, dev_x, dev_labels = pre_process(nlp=nlp, shape=shape, bert_dir=bert_path,
                                                           train_loc=train_loc,
                                                           dev_loc=dev_loc, transformer=transformer_type)

    if transformer_type == 'spacy':
        model = build_model(get_embeddings(nlp.vocab), shape, settings)
    else:
        model = build_model_bert(settings)

    model.fit(
        train_x,
        train_labels,
        validation_data=(dev_x, dev_labels),
        epochs=settings["nr_epoch"],
        batch_size=settings["batch_size"],
        verbose=1
    )

    if not (nlp.path / 'similarity').exists():
        (nlp.path / 'similarity').mkdir()
    print("Saving to", nlp.path / 'similarity')
    weights = model.get_weights()
    # remove the embedding matrix.  We can reconstruct it.
    del weights[1]
    with (nlp.path / 'similarity' / 'spacy_model').open('wb') as file_:
        pickle.dump(weights, file_)
    with (nlp.path / 'similarity' / 'spacy_model_config.json').open('w') as file_:
        file_.write(model.to_json())


def pre_process(nlp, shape, transformer, bert_dir, train_loc, dev_loc):
    train_texts1, train_texts2, train_labels = read_snli(train_loc)
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)

    if transformer == 'spacy':
        print("Processing texts using spacy")

        if os.path.isfile(path=path + "Processed_SNLI/Spacy_Processed/train_x.pkl"):
            print("Pre-Processed train file is found now loading")
            with open(path + 'Processed_SNLI/Spacy_Processed/train_x.pkl', 'rb') as f:
                train_x = pickle.load(f)
        else:
            print("There is no pre-processed file of train_X, Pre-Process will start now")
            train_x = create_dataset(nlp=nlp, texts=train_texts1, hypotheses=train_texts2, num_unk=100,
                                     max_length=shape[0])
            with open(path + 'Processed_SNLI/Spacy_Processed/train_x.pkl', 'wb') as f:
                pickle.dump(train_x, f)

        if os.path.isfile(path=path + "Processed_SNLI/Spacy_Processed/dev_x.pkl"):
            print("Pre-Processed dev file is found now loading")
            with open(path + 'Processed_SNLI/Spacy_Processed/dev_x.pkl', 'rb') as f:
                dev_x = pickle.load(f)
        else:
            print("There is no pre-processed file of dev_X, Pre-Process will start now")
            dev_x = create_dataset(nlp=nlp, texts=dev_texts1, hypotheses=dev_texts2, num_unk=100, max_length=shape[0])
            with open(path + 'Processed_SNLI/Spacy_Processed/dev_x.pkl', 'wb') as f:
                pickle.dump(dev_x, f)
        return train_x, train_labels, dev_x, dev_labels

    else:
        print("Processing texts using bert")
        if os.path.isfile(path=path + "Processed_SNLI/Bert_Processed/train_x.pkl"):
            print("Pre-Processed train file is found now loading")
            with open(path + 'Processed_SNLI/Bert_Processed/train_x.pkl', 'rb') as f:
                train_x = pickle.load(f)
        else:
            print("There is no pre-processed file of train_X, Pre-Process will start now")
            train_sentences = train_texts1 + train_texts2
            vectors_train = extract_features.main(bert_dir, input_file=train_sentences)
            train_x = [np.array(vectors_train[: len(train_texts1)]), np.array(vectors_train[len(train_texts2):])]
            with open(path + 'Processed_SNLI/Bert_Processed/train_x.pkl', 'wb') as f:
                pickle.dump(train_x, f, protocol=pickle.HIGHEST_PROTOCOL)

        if os.path.isfile(path=path + "Processed_SNLI/Bert_Processed/dev_x.pkl"):
            print("Pre-Processed dev file is found now loading")
            with open(path + 'Processed_SNLI/Bert_Processed/dev_x.pkl', 'rb') as f:
                dev_x = pickle.load(f)
        else:
            print("There is no pre-processed file of dev_X, Pre-Process will start now")
            dev_sentences = dev_texts1 + dev_texts2
            vectors_dev = extract_features.main(bert_dir, input_file=dev_sentences)
            dev_x = [np.array(vectors_dev[: len(dev_texts1)]), np.array(vectors_dev[len(dev_texts2):])]
            with open(path + 'Processed_SNLI/Bert_Processed/dev_x.pkl', 'wb') as f:
                pickle.dump(dev_x, f, protocol=pickle.HIGHEST_PROTOCOL)

        return np.asarray(train_x), train_labels, np.asarray(dev_x), dev_labels


def evaluate(dev_loc, shape, bert_path):
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe(KerasSimilarityShim.load(nlp.path / "similarity", nlp, shape[0]))
    total = 0.0
    correct = 0.0
    for text1, text2, label in zip(dev_texts1, dev_texts2, dev_labels):
        doc1 = nlp(text1, disable=['parser', 'tagger', 'ner', 'textcat'])
        doc2 = nlp(text2, disable=['parser', 'tagger', 'ner', 'textcat'])
        sim, _ = doc1.similarity(doc2)
        if sim == KerasSimilarityShim.entailment_types[label.argmax()]:
            correct += 1
        total += 1
    return correct, total


def demo(shape):
    nlp = en_core_web_lg.load()
    nlp.add_pipe(KerasSimilarityShim.load(nlp.path / "similarity", nlp, shape[0]))

    # doc1 = nlp("The king of France is bald.", disable=['parser', 'tagger', 'ner', 'textcat'])
    # doc2 = nlp("France has no king.", disable=['parser', 'tagger', 'ner', 'textcat'])

    doc1 = nlp("option one is much more better",
               disable=['parser', 'tagger', 'ner', 'textcat'])
    doc2 = nlp("option one is worst",
               disable=['parser', 'tagger', 'ner', 'textcat'])

    print("Sentence 1:", doc1)
    print("Sentence 2:", doc2)

    entailment_type, confidence = doc1.similarity(doc2)
    print("Entailment type:", entailment_type, "(Confidence:", confidence, ")")


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


def create_dataset(nlp, texts, hypotheses, num_unk, max_length):
    sents = texts + hypotheses
    # print(sents)
    sents_as_ids = []

    print(len(sents))
    starttime = datetime.datetime.now()
    count = 0

    for sent in sents:
        doc = nlp(sent, disable=['parser', 'tagger', 'ner', 'textcat'])
        word_ids = []
        for i, token in enumerate(doc):
            # i indisi work leri tek tek numaralandırıyor bu sayede max lenght ile karsılastırır.
            # skip odd spaces from tokenizer
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

        count = count + 1
        if count % 50000 == 0:
            print("total sentence: " + str(count) + " Total percent: " + str(count / len(sents)))

    finishtime = datetime.datetime.now()
    totaltime = finishtime - starttime

    print("Total time elapse:" + str(totaltime))
    # text ler ve hipotezleri ayrı ayrı diziler olarak alıyor birinci kısım text - ikinci kısım hipotez
    return [np.array(sents_as_ids[: len(texts)]), np.array(sents_as_ids[len(texts):])]


@plac.annotations(
    mode=("Mode to execute", "positional", None, str, ["train", "evaluate", "demo"]),
    train_loc=("Path to training data", "option", "t", str),
    dev_loc=("Path to development or test data", "option", "s", str),
    max_length=("Length to truncate sentences", "option", "L", int),
    nr_hidden=("Number of hidden units", "option", "H", int),
    dropout=("Dropout level", "option", "d", float),
    learn_rate=("Learning rate", "option", "r", float),
    batch_size=("Batch size for neural network training", "option", "b", int),
    nr_epoch=("Number of training epochs", "option", "e", int),
    entail_dir=(
            "Direction of entailment",
            "option",
            "D",
            str,
            ["both", "left", "right"],
    ),
)
def main(
        mode="train",
        transformer_type='bert',
        bert_location=path + "bert/",
        train_loc=path + "SNLI/snli_train.jsonl",
        dev_loc=path + "SNLI/snli_dev.jsonl",
        test_loc=path + "SNLI/snli_test.jsonl",
        max_length=64,
        nr_hidden=200,
        dropout=0.2,
        learn_rate=0.001,
        batch_size=1024,
        nr_epoch=10,
        entail_dir="both",
):
    shape = (max_length, nr_hidden, 3)
    settings = {
        "lr": learn_rate,
        "dropout": dropout,
        "batch_size": batch_size,
        "nr_epoch": nr_epoch,
        "entail_dir": entail_dir,
    }

    if mode == "train":
        if train_loc == None or dev_loc == None:
            print("Train mode requires paths to training and development data sets.")
            sys.exit(1)
        train(train_loc=train_loc, dev_loc=dev_loc, shape=shape, settings=settings, bert_path=bert_location,
              transformer_type=transformer_type)
    elif mode == "evaluate":
        if dev_loc == None:
            print("Evaluate mode requires paths to test data set.")
            sys.exit(1)
        correct, total = evaluate(test_loc, shape, bert_location)
        print(correct, "/", total, correct / total)
    else:
        demo(shape)


if __name__ == "__main__":
    plac.call(main)
