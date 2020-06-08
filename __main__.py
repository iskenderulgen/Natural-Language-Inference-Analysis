import os
import os.path
import sys

import plac
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from model import esim_bilstm_model, decomp_attention_model
from prediction import SpacyPrediction, BertWordPredict
from prediction_based.bert_encoder import bert_transformer
from pretrained_based.bert_initial_weights import bert_initial_weights_transformer
from pretrained_based.word_vectors import spacy_word_transformer
from utils.utils import read_snli, load_spacy_nlp, attention_visualization

path = "/media/ulgen/Samsung/contradiction_data/"


def train(train_loc, dev_loc, shape, settings, transformer_type, embedding_type):
    train_x, train_labels, dev_x, dev_labels, model = None, None, None, None, None
    if transformer_type == 'glove':  # or 'word2vec' or 'fasttext':
        train_x, train_labels, dev_x, dev_labels, vectors = spacy_word_transformer(path=path, train_loc=train_loc,
                                                                                   dev_loc=dev_loc, shape=shape,
                                                                                   transformer_type=transformer_type)
        model = esim_bilstm_model(vectors=vectors, shape=shape, settings=settings, embedding_type=embedding_type)

    elif transformer_type == 'bert_initial_word':
        train_x, train_labels, dev_x, dev_labels, word_weights = bert_initial_weights_transformer(path=path,
                                                                                                  train_loc=train_loc,
                                                                                                  dev_loc=dev_loc,
                                                                                                  transformer_type=transformer_type)
        model = esim_bilstm_model(vectors=word_weights, shape=shape, settings=settings,
                                  embedding_type=embedding_type)

    elif transformer_type == 'bert_sentence':
        train_x, train_labels, dev_x, dev_labels = bert_transformer(path=path, train_loc=train_loc,
                                                                    dev_loc=dev_loc, feature_type=transformer_type)
        model = esim_bilstm_model(shape=shape, settings=settings, embedding_type=embedding_type, vectors=None)

    else:
        print("Please define transformer method properly")

    model.summary()

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5, restore_best_weights=True)
    history = model.fit(
        train_x,
        train_labels,
        validation_data=(dev_x, dev_labels),
        epochs=settings["nr_epoch"],
        batch_size=settings["batch_size"],
        verbose=2,
        callbacks=[es]
    )

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print('\nhistory dict:', history.history)

    if not os.path.isdir(path + 'similarity'):
        os.mkdir(path + 'similarity')
    print("Saving to", path + 'similarity')

    model.save(path + 'similarity/' + transformer_type + "_" + "model.h5")


def evaluate(dev_loc, shape, transformer_type):
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)
    disabled_pipelines = ['parser', 'tagger', 'ner', 'textcat']
    print("evaluation dataset loaded")
    nlp = load_spacy_nlp(path=path, transformer_type=transformer_type)
    nlp.add_pipe(SpacyPrediction.load(path=path + 'similarity/' + transformer_type + "_" + "model.h5",
                                      max_length=shape[0]))
    total = 0.0
    true_p = 0.0
    for text1, text2, label in zip(dev_texts1, dev_texts2, dev_labels):
        doc1 = nlp(text1, disable=disabled_pipelines)
        doc2 = nlp(text2, disable=disabled_pipelines)
        y_prediction, _, _, _ = doc1.similarity(doc2)
        if y_prediction == SpacyPrediction.entailment_types[label.argmax()]:
            true_p += 1
        total += 1
    print("Entailment Model Accuracy is", true_p / total)

    return true_p, total


def demo(shape, visualization, transformer_type):
    premise = "all i have to say on this issue is that there is actual evidence to support evolution!!"
    hypothesis = "I have to contradict phro and say that the peppered moths do show evidence of evolution. The data may have been insufficient, but evolution did occur. When different alleles are expressed due to external factors, this is evolution."

    if transformer_type == 'glove' or 'fasttext' or 'word2vec':
        nlp = load_spacy_nlp(path=path, transformer_type=transformer_type)
        nlp.add_pipe(SpacyPrediction.load(path=path + 'similarity/' + transformer_type + "_" + "model.h5",
                                          max_length=shape[0]))
        disabled_pipelines = ['parser', 'tagger', 'ner', 'textcat']

        doc1 = nlp(premise, disable=disabled_pipelines)
        doc2 = nlp(hypothesis, disable=disabled_pipelines)

        print("premise:", doc1)
        print("hypothesis   :", doc2)

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

            attention_visualization(tokens1=tokens1, tokens2=tokens2, attention1=attention1, attention2=attention2,
                                    path=path, transformer_type=transformer_type)

    elif transformer_type == 'bert_initial_word':

        entailment_type, confidence, attention1, attention2, sent_tokens = BertWordPredict.predict(
            premise=premise, hypothesis=hypothesis, path=path, transformer_type=transformer_type)
        print("Entailment type:", entailment_type, "(Confidence:", confidence, ")")

        if visualization:
            attention_visualization(tokens1=sent_tokens[0], tokens2=sent_tokens[1], attention1=attention1,
                                    attention2=attention2, path=path, transformer_type=transformer_type)


def main(
        mode="train",
        embedding_type="word",
        transformer_type="bert_initial_word",
        train_loc=path + "SNLI/snli_train.jsonl",
        dev_loc=path + "SNLI/snli_dev.jsonl",
        test_loc=path + "SNLI/snli_test.jsonl",
        max_length=64,  # 64 for word based #1024 for bert_dependencies sentence
        nr_hidden=300,
        dropout=0.2,
        learn_rate=0.0004,
        batch_size=32,
        nr_epoch=20,
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
              transformer_type=transformer_type, embedding_type=embedding_type)
    elif mode == "evaluate":
        if dev_loc is None:
            print("Evaluate mode requires paths to test data set.")
            sys.exit(1)
        evaluate(test_loc, shape, transformer_type=transformer_type)
    else:
        demo(shape, transformer_type=transformer_type, visualization=attention_visualization)


if __name__ == "__main__":
    plac.call(main)
