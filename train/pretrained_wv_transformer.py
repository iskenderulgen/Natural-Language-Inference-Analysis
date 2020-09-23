"""
This code converts premises and hypothesis using pre trained word weights. Currently it supports 3
(word2vec, glove, fasttext) word weights. All are pruned to 685k unique vectors. Pruning conducted
based on spacy's init module. Unique vector size referred from original spacy's glove weight size.
"""
import argparse
import datetime
import os
import pickle
import cupy as cp
import numpy as np
import plac
import yaml
from keras.callbacks import EarlyStopping

from utils.utils import read_snli, load_spacy_nlp
from models.esim import esim_bilstm_model
from models.decomposable_attention import decomposable_attention_model
import matplotlib.pyplot as plt


with open("/home/ulgen/Documents/Python_Projects/Contradiction/configurations.yaml", 'r') as stream:
    try:
        configs = (yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)


parser = argparse.ArgumentParser()
parser.add_argument("--transformer_path", type=str, default=None,
                    help="Transformer model path that will convert text to vectors")
parser.add_argument("processed_path",type=str, default=None,
                    help="")
args = parser.parse_args()


def create_dataset_ids(nlp, premises, hypothesis, num_unk, max_length):
    """This section creates id matrix of the input tokens"""

    sentences = premises + hypothesis
    sentences_as_ids = []

    print("Total number of premises and hypothesis to be processed = ", len(sentences))
    start_time = datetime.datetime.now()
    processed_sent_count = 0

    for sent in sentences:
        doc = nlp(sent, disable=['parser', 'tagger', 'ner', 'textcat'])
        word_ids = []
        for i, token in enumerate(doc):
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
        sentences_as_ids.append(word_id_vec)

        processed_sent_count = processed_sent_count + 1
        if processed_sent_count % 5000 == 0:
            print("processed Sentence:", str(processed_sent_count),
                  "Processed Percentage:", str(round(processed_sent_count / len(sentences), 4) * 100))

    finish_time = datetime.datetime.now()
    print("Total time spent to create token ID's of sentences: ", (finish_time - start_time))

    return [np.array(sentences_as_ids[: len(premises)]), np.array(sentences_as_ids[len(premises):])]


def get_embeddings(vocab, nr_unk):
    # the extra +1 is for a zero vector representing sentence-final padding
    num_vectors = max(lex.rank for lex in vocab) + 2

    # create random vectors for OOV tokens
    oov = np.random.normal(size=(nr_unk, vocab.vectors_length))
    oov = oov / oov.sum(axis=1, keepdims=True)

    vectors = np.zeros((num_vectors + nr_unk, vocab.vectors_length), dtype="float32")
    vectors[1: (nr_unk + 1), ] = oov
    for lex in vocab:
        if lex.has_vector and lex.vector_norm > 0:
            vectors[nr_unk + lex.rank + 1] = cp.asnumpy(lex.vector / lex.vector_norm)

    print("Extracting embeddings is finished")

    return vectors


def spacy_word_transformer(transformer_path, processed_path, train_loc, dev_loc, max_length, transformer_type, nr_unk):
    print("Starting to pre-process using spacy. Transformer type is ", transformer_type)

    nlp = load_spacy_nlp(transformer_path=transformer_path, transformer_type=transformer_type)

    train_texts1, train_texts2, train_labels = read_snli(train_loc)
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)

    if not os.path.isdir(processed_path):
        print("Processed_SNLI directory is not exist, it's now created")
        os.mkdir(processed_path)

    if os.path.isfile(path=processed_path + transformer_type + "/train_x.pkl"):
        print(transformer_type, "based Pre-Processed train file is found now loading...")
        with open(processed_path + transformer_type + "/train_x.pkl", "rb") as f:
            train_x = pickle.load(f)
    else:
        print(transformer_type, "based pre-processed file of train_X isn't exist, Pre-Process will start now")
        train_x = create_dataset_ids(nlp=nlp, premises=train_texts1, hypothesis=train_texts2, num_unk=nr_unk,
                                     max_length=max_length)
        with open(processed_path + transformer_type + "/train_x.pkl", "wb") as f:
            pickle.dump(train_x, f)

    if os.path.isfile(path=processed_path + transformer_type + "/dev_x.pkl"):
        print(transformer_type, "based Pre-Processed dev file is found now loading...")
        with open(processed_path + transformer_type + "/dev_x.pkl", "rb") as f:
            dev_x = pickle.load(f)
    else:
        print(transformer_type, "based pre-processed file of dev_X isn't exist, Pre-Process will start now")
        dev_x = create_dataset_ids(nlp=nlp, premises=dev_texts1, hypothesis=dev_texts2, num_unk=nr_unk,
                                   max_length=max_length)
        with open(processed_path + transformer_type + "/dev_x.pkl", "wb") as f:
            pickle.dump(dev_x, f)

    if os.path.isfile(path=processed_path + transformer_type + "/weights.pkl"):
        print(transformer_type, "weights matrix already extracted, now loading...")
        with open(processed_path + transformer_type + "/weights.pkl", "rb") as f:
            vectors = pickle.load(f)
    else:
        print(transformer_type, " weight matrix is not found, now extracting...")
        vectors = get_embeddings(vocab=nlp.vocab, nr_unk=nr_unk)
        with open(processed_path + transformer_type + "/weights.pkl", "wb") as f:
            pickle.dump(vectors, f)

    return train_x, train_labels, dev_x, dev_labels, vectors


def train_model(model_save_path, model_type, max_length, batch_size, nr_epoch,
                nr_hidden, nr_class, learning_rate, embedding_type,
                train_x, train_labels, dev_x, dev_labels, vectors):
    model = None

    if model_type == "esim":

        model = esim_bilstm_model(vectors=vectors, max_length=max_length, nr_hidden=nr_hidden,
                                  nr_class=nr_class, learning_rate=learning_rate, embedding_type=embedding_type)

    elif model_type == "decomposable_attention":

        model = decomposable_attention_model(vectors=vectors, max_length=max_length,
                                             nr_hidden=nr_hidden, nr_class=nr_class,
                                             learning_rate=learning_rate, embedding_type=embedding_type)

    model.summary()

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=4, restore_best_weights=True)

    history = model.fit(
        train_x,
        train_labels,
        validation_data=(dev_x, dev_labels),
        epochs=nr_epoch,
        batch_size=batch_size,
        verbose=1,
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

    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)
    print("Saving to", model_save_path)

    model.save(model_save_path + model_type + "/model.h5")


def main():

    train_x, train_labels, dev_x, dev_labels, vectors = spacy_word_transformer(transformer_path=, processed_path=,
                                                                               train_loc=, dev_loc=, max_length=,
                                                                               transformer_type=, nr_unk=)

    train_model(model_save_path=, model_type=, max_length=, batch_size=, nr_epoch=,
                nr_hidden=, nr_class=, learning_rate=, embedding_type=,
                train_x=, train_labels=, dev_x=, dev_labels=, vectors=)

if __name__ == "__main__":
    plac.call(main)
