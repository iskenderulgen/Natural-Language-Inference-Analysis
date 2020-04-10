import os
import os.path
import sys

import plac

from Transformers.bert_sentence_based import bert_sentence_transformer
from Transformers.bert_word_based import bert_word_based_transformer
from Transformers.glove_word_based import glove_word_transformer
from Transformers.spacy_based import spacy_word_transformer
from Transformers.utils import read_snli, load_spacy_nlp, attention_visualization, set_keras_backend
from model import decomp_attention_model, esim_bilstm_model
from prediction import SpacyPrediction, BertWordPredict

path = "/media/ulgen/Samsung/contradiction_data/"

set_keras_backend("tensorflow")


def train(train_loc, dev_loc, shape, settings, transformer_type, train_type):
    train_x, train_labels, dev_x, dev_labels, model = None, None, None, None, None
    if transformer_type == 'spacy':
        train_x, train_labels, dev_x, dev_labels, vectors = spacy_word_transformer(path=path, train_loc=train_loc,
                                                                                   dev_loc=dev_loc, shape=shape,
                                                                                   transformer_type=transformer_type)
        model = decomp_attention_model(vectors=vectors, shape=shape, settings=settings, train_type=train_type)

    elif transformer_type == 'glove':
        train_x, train_labels, dev_x, dev_labels, vectors = glove_word_transformer(path=path, train_loc=train_loc,
                                                                                   dev_loc=dev_loc, shape=shape,
                                                                                   transformer_type=transformer_type)
        model = decomp_attention_model(vectors=vectors, shape=shape, settings=settings, train_type=train_type)

    elif transformer_type == 'bert_word_based':
        train_x, train_labels, dev_x, dev_labels, word_weights = bert_word_based_transformer(path=path,
                                                                                             train_loc=train_loc,
                                                                                             dev_loc=dev_loc,
                                                                                             transformer_type=transformer_type)
        model = decomp_attention_model(vectors=word_weights, shape=shape, settings=settings,train_type=train_type)

    elif transformer_type == 'bert_sentence':
        train_x, train_labels, dev_x, dev_labels = bert_sentence_transformer(path=path, train_loc=train_loc,
                                                                             dev_loc=dev_loc)
        model = decomp_attention_model(shape=shape, settings=settings, train_type=train_type, vectors=None)

    else:
        print("Please define transformer method properly")

    model.summary()

    model.fit(
        train_x,
        train_labels,
        validation_data=(dev_x, dev_labels),
        epochs=settings["nr_epoch"],
        batch_size=settings["batch_size"],
        verbose=1
    )

    if not os.path.isdir(path + 'similarity'):
        os.mkdir(path + 'similarity')
    print("Saving to", path + 'similarity')

    model.save(path + 'similarity/' + train_type + "_" + "model.h5")


def evaluate(dev_loc, shape):
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)
    print("evaluation dataset loaded")
    nlp = load_spacy_nlp()
    nlp.add_pipe(SpacyPrediction.load(path=path + "similarity/", max_length=shape[0]))
    total = 0.0
    correct = 0.0
    for text1, text2, label in zip(dev_texts1, dev_texts2, dev_labels):
        doc1 = nlp(text1, disable=['parser', 'tagger', 'ner', 'textcat'])
        doc2 = nlp(text2, disable=['parser', 'tagger', 'ner', 'textcat'])
        y_prediction, _ = doc1.similarity(doc2)
        if y_prediction == SpacyPrediction.entailment_types[label.argmax()]:
            correct += 1
        total += 1
    return correct, total


def demo(shape, type, visualization):
    hypothesis = "A man inspects the uniform of a figure in some East Asian country."
    premise = "The man is sleeping."

    if type == 'spacy':
        nlp = load_spacy_nlp()
        nlp.add_pipe(SpacyPrediction.load(path=path + "similarity/", max_length=shape[0]))
        disabled_pipelines = ['parser', 'tagger', 'ner', 'textcat']

        doc1 = nlp(hypothesis, disable=disabled_pipelines)
        doc2 = nlp(premise, disable=disabled_pipelines)

        print("hypothesis:", doc1)
        print("premise   :", doc2)

        entailment_type, confidence, attention1, attention2 = doc1.similarity(doc2)
        print("Entailment type:", entailment_type, "(Confidence:", confidence, ")")

        if visualization:
            def sents_to_words(doc):
                words = []
                for token in doc:
                    words.append(token.text)
                return words

            tokens1 = sents_to_words(doc=doc1)
            tokens2 = sents_to_words(doc=doc2)

            attention_visualization(tokens1=tokens1, tokens2=tokens2, attention1=attention1, attention2=attention2)

    elif type == 'bert':

        entailment_type, confidence, attention1, attention2, sent_tokens = BertWordPredict.predict(
            hypothesis=hypothesis, premise=premise, path=path)
        print("Entailment type:", entailment_type, "(Confidence:", confidence, ")")

        attention_visualization(tokens1=sent_tokens[0], tokens2=sent_tokens[1], attention1=attention1,
                                attention2=attention2)


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
)
def main(
        mode="train",
        train_type="word",
        transformer_type='bert_word_based',
        train_loc=path + "SNLI/snli_train.jsonl",
        dev_loc=path + "SNLI/snli_dev.jsonl",
        test_loc=path + "SNLI/snli_test.jsonl",
        max_length=50,  # 48 for word based #1024 for bert sentence
        nr_hidden=200,  # 200
        dropout=0.2,
        learn_rate=0.001,  # 0.001
        batch_size=1024,  # 100 for ESIM
        nr_epoch=7,
        attention_visualization=True):
    shape = (max_length, nr_hidden, 3)
    settings = {
        "lr": learn_rate,
        "dropout": dropout,
        "batch_size": batch_size,
        "nr_epoch": nr_epoch,
    }

    if mode == "train":
        if train_loc is None or dev_loc is None:
            print("Train mode requires paths to training and development data sets.")
            sys.exit(1)
        train(train_loc=train_loc, dev_loc=dev_loc, shape=shape, settings=settings,
              transformer_type=transformer_type, train_type=train_type)
    elif mode == "evaluate":
        if dev_loc is None:
            print("Evaluate mode requires paths to test data set.")
            sys.exit(1)
        correct, total = evaluate(test_loc, shape)
        print(correct, "/", total, correct / total)
    else:
        demo(shape, type=transformer_type, visualization=attention_visualization)


if __name__ == "__main__":
    plac.call(main)
