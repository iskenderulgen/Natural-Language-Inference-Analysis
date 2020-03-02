import importlib
import os
import os.path
import pickle
import sys

import numpy as np
import plac
from keras import backend as k_backend

from Transformers.bert_sentence_based import bert_sentence_transformer
from Transformers.bert_word_based import bert_word_based_transformer
from Transformers.spacy_based import spacy_word_transformer
from Transformers.utils import read_snli, load_spacy_nlp
from keras_decomposable_attention import build_model_word_based, build_model_sentence_based
from spacy_hook import get_embeddings, KerasSimilarityShim

path = "/media/ulgen/Samsung/contradiction_data/data/"


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


def train(train_loc, dev_loc, shape, settings, transformer_type):
    train_x, train_labels, dev_x, dev_labels, model = None, None, None, None, None
    if transformer_type == 'spacy':
        train_x, train_labels, dev_x, dev_labels, vectors = spacy_word_transformer(path=path, train_loc=train_loc,
                                                                                   dev_loc=dev_loc, shape=shape,
                                                                                   transformer_type=transformer_type)
        model = build_model_word_based(vectors=vectors, shape=shape, settings=settings)

    elif transformer_type == 'bert_word_based':
        train_x, train_labels, dev_x, dev_labels, word_weights = bert_word_based_transformer(path=path,
                                                                                             train_loc=train_loc,
                                                                                             dev_loc=dev_loc,
                                                                                             transformer_type=transformer_type)
        model = build_model_word_based(vectors=word_weights, shape=shape, settings=settings)

    elif transformer_type == 'bert_sentence':
        train_x, train_labels, dev_x, dev_labels = bert_sentence_transformer(path=path, train_loc=train_loc,
                                                                             dev_loc=dev_loc)
        model = build_model_sentence_based(shape=shape, settings=settings)

    else:
        print("Please define transformer method properly")

    model.fit(
        train_x,
        train_labels,
        validation_data=(dev_x, dev_labels),
        epochs=settings["nr_epoch"],
        batch_size=settings["batch_size"],
        verbose=1
    )

    if not os.path.isdir(path + 'similarity'):
        os.mkdir(path+'similarity')
    print("Saving to", path + 'similarity')
    weights = model.get_weights()
    # remove the embedding matrix.  We can reconstruct it.
    del weights[1]
    with (path + 'similarity/' + 'spacy_model', 'wb') as file_:
        pickle.dump(weights, file_)
    with (path + 'similarity/' + 'spacy_model_config.json', 'w') as file_:
        file_.write(model.to_json())


def evaluate(dev_loc, shape, bert_path):
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)

    nlp = load_spacy_nlp()
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
    nlp = load_spacy_nlp()
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
        transformer_type='bert_word_based',
        bert_location=path + "bert/",
        train_loc=path + "SNLI/snli_train.jsonl",
        dev_loc=path + "SNLI/snli_dev.jsonl",
        test_loc=path + "SNLI/snli_test.jsonl",
        max_length=64,  # 64 for word based
        nr_hidden=400,  # 200
        dropout=0.2,
        learn_rate=0.0001,  # 0.001
        batch_size=128,
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
        if train_loc is None or dev_loc is None:
            print("Train mode requires paths to training and development data sets.")
            sys.exit(1)
        train(train_loc=train_loc, dev_loc=dev_loc, shape=shape, settings=settings,
              transformer_type=transformer_type)
    elif mode == "evaluate":
        if dev_loc is None:
            print("Evaluate mode requires paths to test data set.")
            sys.exit(1)
        correct, total = evaluate(test_loc, shape, bert_location)
        print(correct, "/", total, correct / total)
    else:
        demo(shape)


if __name__ == "__main__":
    plac.call(main)
