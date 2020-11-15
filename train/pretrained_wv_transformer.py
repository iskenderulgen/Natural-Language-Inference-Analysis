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
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping
from utilities.utils import read_nli, load_spacy_nlp, load_configurations
from models.esim import esim_bilstm_model
from models.decomposable_attention import decomposable_attention_model

configs = load_configurations()

parser = argparse.ArgumentParser()
parser.add_argument("--transformer_type", type=str, default="glove",
                    help="Type of the transformer which will convert texts in to word-ids. Currently three types "
                         "are supported.Here the types as follows 'glove' -  'fasttext' - 'word2vec'."
                         "Pick one you'd like to transform into")

parser.add_argument("--embedding_type", type=str, default="word",
                    help="For word embedding base models use 'word' keyword,"
                         "For sentence embedding base models use 'sentence' keyword. "
                         "Required embedding layer will be triggered based on selection")

parser.add_argument("--model_type", type=str, default="esim",
                    help="Type of the model that will be trained. "
                         "for ESIM model type 'esim' "
                         "for decomposable attention model type 'decomposable_attention'. ")

parser.add_argument("--transformer_path", type=str, default=configs["transformer_paths"],
                    help="Main transformer model path which will convert the text in to word-ids and vectors. "
                         "transformer path has four sub paths, load_nlp module will carry out the sub model paths"
                         "based on transformer_type selection")

parser.add_argument("--train_loc", type=str, default=configs["nli_set_train"],
                    help="Train data location which will be processed via transformers and be saved to processed_path "
                         "location")

parser.add_argument("--dev_loc", type=str, default=configs["nli_set_dev"],
                    help="Train dev data location which will be used to measure train accuracy while training model,"
                         "files will be processed using transformer and saved to processed path")

parser.add_argument("--max_length", type=str, default=configs["max_length"],
                    help="Max length of the sentences,longer sentences will be pruned and shorter ones will be zero"
                         "padded. Remember longer sentences means longer sequences to train. Select best length based"
                         "on your rig.")

parser.add_argument("--nr_unk", type=int, default=configs["nr_unk"],
                    help="number of unknown vectors which will be used for padding the short sentences to desired"
                         "length.Nr unknown vectors will be created using random module")

parser.add_argument("--processed_path", type=str, default=configs["processed_nli"],
                    help="Path where the transformed texts will be saved to as word-ids. Word-id matrix will be used"
                         "in embedding layer as a look-up table.")

parser.add_argument("--model_save_path", type=str, default=configs["model_paths"],
                    help="The path where trained NLI model will be saved.")

parser.add_argument("--batch_size", type=int, default=configs["batch_size"],
                    help="Batch size of model, it represents the amount of data the model will train for each pass.")

parser.add_argument("--nr_epoch", type=int, default=configs["nr_epoch"],
                    help="Total number of times that model will iterate trough the data.")

parser.add_argument("--nr_hidden", type=int, default=configs["nr_hidden"],
                    help="Hidden neuron size of the model")

parser.add_argument("--nr_class", type=int, default=configs["nr_class"],
                    help="Number of class that will model classify the data into. Also represents the last layer of"
                         "the model.")

parser.add_argument("--learning_rate", type=float, default=configs["learn_rate"],
                    help="Learning rate parameter that represent the constant which will be multiplied with the data"
                         "in each back propagation")

parser.add_argument("--result_path", type=str, default=configs["results"],
                    help="path of the file where trained model loss and accuracy graphs will be saved.")

parser.add_argument("--early_stopping", type=int, default=configs["early_stopping"],
                    help="early stopping parameter for model, which stops training when reaching best accuracy.")
args = parser.parse_args()


def create_dataset_ids(nlp, premises, hypothesis, num_unk, max_length):
    """
    This function takes hypothesis and premises as list and converts them to word-ids matrix of the input tokens
    based on lookup table which will be converted to vectors in the embedding layer of the training model.
    :param nlp: transformer model which has the lookup table
    :param premises: opinion sentence
    :param hypothesis: opinion sentence
    :param num_unk: unknown word count that will be filled with norm_random vectors
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :return: returns word ids as a list.
    """

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
    """
    This function takes the embeddings vectors from nlp object and adds the unknown word vectors to it. Later it
    saves the whole new vector object to disk as okl file. It will be used in embedding layer to match the word ids
    with corresponding vectors.
    :param vocab: nlp vocabulary object that has words and vectors.
    :param nr_unk: unknown word vector size that will be used to create random_norm vectors for out of vocabulary words.
    :return: returns new vector table.
    """

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


def spacy_word_transformer(transformer_path, transformer_type, train_loc, dev_loc, max_length, nr_unk, processed_path):
    """
    This function reads NLI sets and processes them trough the functions above. Takes sentences as list and transforms
    them in to word-id matrix. This word_id matrix will be then saved to disk as pkl file to be read and used in
    embedding layer of the madel. Currently this method supports glove - fasttext and word2vec pretrained weights.
    :param transformer_path: path of the transformer nlp object.
    :param transformer_type: type of the transformer glove - fasttext or word2vec.
    :param train_loc: training NLI jsonl date location.
    :param dev_loc: dev NLI jsonl date location
    :param max_length: max length of the sentence. Longer ones will be pruned shorter ones will be padded.
    :param nr_unk: number of unknown word size. Random weight will be created based on this unk word size
    :param processed_path: path where the processed files will be based.
    :return: returns train - dev set and corresponding labels with word weights.
    """

    print("Starting to pre-process using spacy. Transformer type is ", transformer_type)

    nlp = load_spacy_nlp(transformer_path=transformer_path, transformer_type=transformer_type)

    train_texts1, train_texts2, train_labels = read_nli(train_loc)
    dev_texts1, dev_texts2, dev_labels = read_nli(dev_loc)

    if not os.path.isdir(processed_path):
        print("Processed_nli directory is not exist, it's now created")
        os.mkdir(processed_path)

    if os.path.isfile(path=processed_path + "train_x.pkl"):
        print(transformer_type, "based Pre-Processed train file is found now loading...")
        with open(processed_path + "train_x.pkl", "rb") as f:
            train_x = pickle.load(f)
    else:
        print(transformer_type, "based pre-processed file of train_nli isn't exist, pre process starts now")
        train_x = create_dataset_ids(nlp=nlp, premises=train_texts1, hypothesis=train_texts2, num_unk=nr_unk,
                                     max_length=max_length)
        with open(processed_path + "train_x.pkl", "wb") as f:
            pickle.dump(train_x, f)

    if os.path.isfile(path=processed_path + "dev_x.pkl"):
        print(transformer_type, "based pre processed dev file is found, now loading...")
        with open(processed_path + "dev_x.pkl", "rb") as f:
            dev_x = pickle.load(f)
    else:
        print(transformer_type, "based pre processed file of train_dev isn't exist, pre process will start now.")
        dev_x = create_dataset_ids(nlp=nlp, premises=dev_texts1, hypothesis=dev_texts2, num_unk=nr_unk,
                                   max_length=max_length)
        with open(processed_path + "dev_x.pkl", "wb") as f:
            pickle.dump(dev_x, f)

    if os.path.isfile(path=transformer_path[transformer_type] + "weights.pkl"):
        print(transformer_type, "weights matrix already extracted, now loading...")
        with open(transformer_path[transformer_type] + "weights.pkl", "rb") as f:
            vectors = pickle.load(f)
    else:
        print(transformer_type, " weight matrix is not found, now extracting...")
        vectors = get_embeddings(vocab=nlp.vocab, nr_unk=nr_unk)
        with open(transformer_path[transformer_type] + "weights.pkl", "wb") as f:
            pickle.dump(vectors, f)

    return train_x, train_labels, dev_x, dev_labels, vectors


def train_model(model_save_path, model_type, max_length, batch_size, nr_epoch,
                nr_hidden, nr_class, learning_rate, embedding_type, early_stopping,
                train_x, train_labels, dev_x, dev_labels, vectors, result_path):
    """
    Model will be trained in this function. Currently it supports ESIM and Decomposable Attention models.
    :param model_save_path: path where the model will be saved as h5 file.
    :param model_type: type of the model. either ESIM or Decomposable attention.
    :param max_length: max length of the sentence / sequence.
    :param batch_size: size of the train data will be feed forwarded on each iteration.
    :param nr_epoch: total number of times the model iterates trough all the training data.
    :param nr_hidden: Hidden neuron size of the model
    :param nr_class: number of classed that model will classify into. Also the last layer of the model.
    :param learning_rate: constant rate that will be used on each back propagation.
    :param embedding_type: definition of the embeddings for the model. For word embedding based model, 'word' keyword,
    for sentence based model 'sentence' should be selected.
    :param early_stopping: parameter that stops the training when the validation accuracy cant go higher.
    :param train_x: training data.
    :param train_labels: training labels.
    :param dev_x: developer data
    :param dev_labels: developer labels
    :param vectors: embedding vectors of the words.
    :param result_path: path where accuracy and loss graphs will be saved along with the model history.
    :return: None
    """

    model = None

    if model_type == "esim":

        model = esim_bilstm_model(vectors=vectors, max_length=max_length, nr_hidden=nr_hidden,
                                  nr_class=nr_class, learning_rate=learning_rate, embedding_type=embedding_type)

    elif model_type == "decomposable_attention":

        model = decomposable_attention_model(vectors=vectors, max_length=max_length,
                                             nr_hidden=nr_hidden, nr_class=nr_class,
                                             learning_rate=learning_rate, embedding_type=embedding_type)

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                       patience=early_stopping, restore_best_weights=True)

    history = model.fit(
        train_x,
        train_labels,
        validation_data=(dev_x, dev_labels),
        epochs=nr_epoch,
        batch_size=batch_size,
        verbose=1,
        callbacks=[es]
    )

    if not os.path.isdir(model_save_path[model_type]):
        os.mkdir(model_save_path[model_type])
    print("Saving to", model_save_path[model_type])

    model.save(model_save_path[model_type] + "model.h5")

    print('\n model history:', history.history)

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    with open(result_path + 'result_history.txt', 'w') as file:
        file.write(str(history.history))

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(result_path + 'accuracy.png', bbox_inches='tight')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(result_path + 'loss.png', bbox_inches='tight')
    plt.show()


def main():
    train_x, train_labels, dev_x, dev_labels, vectors = spacy_word_transformer(transformer_path=args.transformer_path,
                                                                               processed_path=args.processed_path,
                                                                               train_loc=args.train_loc,
                                                                               dev_loc=args.dev_loc,
                                                                               max_length=args.max_length,
                                                                               transformer_type=args.transformer_type,
                                                                               nr_unk=args.nr_unk)

    train_model(model_save_path=args.model_save_path,
                model_type=args.model_type,
                max_length=args.max_length,
                batch_size=args.batch_size,
                nr_epoch=args.nr_epoch,
                nr_hidden=args.nr_hidden,
                nr_class=args.nr_class,
                learning_rate=args.learning_rate,
                embedding_type=args.embedding_type,
                early_stopping=args.early_stopping,
                train_x=train_x,
                train_labels=train_labels,
                dev_x=dev_x,
                dev_labels=dev_labels,
                vectors=vectors,
                result_path=args.result_path)


if __name__ == "__main__":
    plac.call(main)
