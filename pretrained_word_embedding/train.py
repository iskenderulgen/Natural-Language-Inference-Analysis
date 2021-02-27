"""
This code converts premises and hypothesis using pre trained word weights. Currently it supports 3
(word2vec, glove, fasttext) word weights. All are pruned to 685k unique vectors. Pruning conducted
based on spacy's init module. Unique vector size referred from original spacy's glove weight size.
"""
import argparse
import datetime
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import plac
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

import utilities
from models.decomposable_attention import decomposable_attention_model
from models.esim import esim_bilstm_model
from utilities.utils import read_nli, load_spacy_nlp, load_configurations

utilities.utils.set_memory_growth()
configs = load_configurations()
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="decomposable_attention",
                    help="Type of the model architecture that the model is trained on. This parameter also carries the "
                         "model save path information hence this is used for both defining architecture and carrying "
                         "path information."
                         "for ESIM model use 'esim'"
                         "for Decomposable Attention model use 'decomposable_attention'.")

parser.add_argument("--transformer_type", type=str, default="glove",
                    help="Type of the transformer which will convert texts in to word-ids. Also carries the path "
                         "information of transformer object. Currently three types are supported.Here the types as"
                         " follows 'glove' -  'fasttext' - 'word2vec' - 'ontonotes5'. Pick one you'd like to"
                         " transform into")

parser.add_argument("--train_loc", type=str, default=configs["nli_set_train"],
                    help="Train data location which will be processed with NLP object and fed in to downstream task.")

parser.add_argument("--dev_loc", type=str, default=configs["nli_set_dev"],
                    help="Dev data location which will be used to measure train accuracy while training model,"
                         "files will be processed with NLP object and fed in to downstream task.")

parser.add_argument("--test_loc", type=str, default=configs["nli_set_test"],
                    help="Test data location which will be used to measure model accuracy after the training model,"
                         "files will be processed with NLP object and fed in to downstream task.")

parser.add_argument("--max_length", type=str, default=configs["max_length"],
                    help="Max length of the sentences, longer sentences will be pruned and shorter ones will be zero"
                         "padded. longer sentences mean longer sequences to train. Pick best length based on your rig.")

parser.add_argument("--nr_unk", type=int, default=configs["nr_unk"],
                    help="number of unknown vectors which will be used for padding the short sentences to desired"
                         "length. Nr unknown vectors will be created using random module")

parser.add_argument("--processed_path", type=str, default=configs["processed_nli"],
                    help="Path where the transformed texts will be saved as word-ids. Word-id matrix will be used"
                         "in embedding layer to retrieve word weights from lookup table.")

parser.add_argument("--batch_size", type=int, default=configs["batch_size"],
                    help="Batch size of model, it represents the amount of data the model will train for each pass.")

parser.add_argument("--nr_epoch", type=int, default=configs["nr_epoch"],
                    help="Total number of times that model will iterate trough whole data.")

parser.add_argument("--nr_hidden", type=int, default=configs["nr_hidden"],
                    help="Hidden neuron size of the model")

parser.add_argument("--nr_class", type=int, default=configs["nr_class"],
                    help="Number of classes that will model classify the data into. Also represents the last layer of"
                         "the model.")

parser.add_argument("--learning_rate", type=float, default=configs["learn_rate"],
                    help="Learning rate parameter which will update the weights on back propagation")

parser.add_argument("--early_stopping", type=int, default=configs["early_stopping"],
                    help="early stopping parameter for model, which stops training when reaching best accuracy.")

parser.add_argument("--result_path", type=str, default=configs["results"],
                    help="path of the folder where trained model loss and accuracy graphs will be saved.")
args = parser.parse_args()


def convert_examples_to_features(nlp, premises, hypothesis, num_unk, max_length):
    """
    This function takes hypothesis and premises as list and converts them to word-ids matrix based on lookup table.
    Extracted word-ids will be converted to vectors in the embedding layer of the training model.
    :param nlp: transformer object that will tokenize and convert tokens to word-ids.
    :param premises: opinion sentence.
    :param hypothesis: opinion sentence.
    :param num_unk: unknown word count that will be filled with norm_random vectors.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :return: returns word-ids as a list.
    """

    sentences = premises + hypothesis
    print("Total number of premises and hypothesis to be processed:", len(sentences))
    start_time = datetime.datetime.now()
    processed_sent_count = 0
    sentences_as_ids = []

    for sent in sentences:
        doc = nlp(sent)
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

        word_id_vec = np.zeros(max_length, dtype="int")
        clipped_len = min(max_length, len(word_ids))
        word_id_vec[:clipped_len] = word_ids[:clipped_len]
        sentences_as_ids.append(word_id_vec)

        processed_sent_count = processed_sent_count + 1
        if processed_sent_count % 5000 == 0:
            print("processed Sentence: %d Processed Percentage: %.2f" %
                  (processed_sent_count, processed_sent_count / len(sentences) * 100))

    print("Total time spent to create token ID's of sentences:", (datetime.datetime.now() - start_time))

    return [np.array(sentences_as_ids[: len(premises)]), np.array(sentences_as_ids[len(premises):])]


def get_embeddings(vocab, nr_unk):
    """
    This function takes the embeddings from nlp object and adds random weights for unknown words. Later it saves
    the whole new vector object to disk as pkl file. It will be used in embedding layer to match the word ids with
    corresponding vectors.
    :param vocab: nlp vocabulary object.
    :param nr_unk: unknown word vector size that will be used to create random_norm vectors for out of vocabulary words.
    :return: returns new vector table.
    """

    # the extra +1 is for a zero vector representing sentence-final padding
    # num_vectors = max(lex.rank for lex in vocab) + 2
    num_vectors = len(vocab.vectors) + 2
    # create random vectors for OOV tokens
    oov = np.random.normal(size=(nr_unk, vocab.vectors_length))
    oov = oov / oov.sum(axis=1, keepdims=True)

    vectors = np.zeros((num_vectors + nr_unk, vocab.vectors_length), dtype="float32")
    vectors[1: (nr_unk + 1), ] = oov
    for lex in vocab:
        if lex.has_vector and lex.vector_norm > 0:
            vectors[nr_unk + lex.rank + 1] = np.asarray(lex.vector / lex.vector_norm)

    print("Extracting embeddings is finished")

    return vectors


def spacy_word_transformer(transformer_type, train_loc, dev_loc, test_loc, max_length, nr_unk, processed_path):
    """
    This function reads NLI sets and processes them with NLP object. Takes sentences as list and transforms them into
    word-id matrix. This word_id matrix will be then saved to disk as pkl file to be read and used in embedding layer
    of the madel. Currently this method supports glove - fasttext and word2vec pretrained weights.
    :param transformer_type: type of the transformer, glove - fasttext or word2vec.
    :param train_loc: NLI train dataset location.
    :param dev_loc: NLI dev dataset location.
    :param test_loc: NLI test data location.
    :param max_length: max length of the sentence. Longer ones will be pruned shorter ones will be padded.
    :param nr_unk: number of unknown word size. Random weights will be created based on this unk word size.
    :param processed_path: path where the processed files will be based.
    :return: train - dev set as word-ids matrix and corresponding labels.
    """

    print("Starting to pre-process using spacy. Transformer type is: ", transformer_type)

    nlp = load_spacy_nlp(configs=configs, transformer_type=transformer_type)
    train_texts1, train_texts2, train_labels = read_nli(train_loc)
    dev_texts1, dev_texts2, dev_labels = read_nli(dev_loc)
    test_texts1, test_texts2, test_labels = read_nli(test_loc)

    if not os.path.isdir(processed_path):
        print("Processed nli directory is not exist, it's now created")
        os.mkdir(processed_path)

    if os.path.isfile(path=processed_path + "train_x.pkl"):
        print(transformer_type, "based Pre-Processed train file is found now loading...")
        with open(processed_path + "train_x.pkl", "rb") as f:
            train_x = pickle.load(f)
    else:
        print(transformer_type, "based pre-processed file of train_nli isn't exist, pre process starts now")
        train_x = convert_examples_to_features(nlp=nlp, premises=train_texts1, hypothesis=train_texts2, num_unk=nr_unk,
                                               max_length=max_length)
        with open(processed_path + "train_x.pkl", "wb") as f:
            pickle.dump(train_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(path=processed_path + "dev_x.pkl"):
        print(transformer_type, "based pre processed dev file is found, now loading...")
        with open(processed_path + "dev_x.pkl", "rb") as f:
            dev_x = pickle.load(f)
    else:
        print(transformer_type, "based pre processed file of train_dev isn't exist, pre process will start now.")
        dev_x = convert_examples_to_features(nlp=nlp, premises=dev_texts1, hypothesis=dev_texts2, num_unk=nr_unk,
                                             max_length=max_length)
        with open(processed_path + "dev_x.pkl", "wb") as f:
            pickle.dump(dev_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(path=processed_path + "test_x.pkl"):
        print(transformer_type, "based pre processed test file is found, now loading...")
        with open(processed_path + "test_x.pkl", "rb") as f:
            test_x = pickle.load(f)
    else:
        print(transformer_type, "based pre processed file of test data isn't exist, pre process will start now.")
        test_x = convert_examples_to_features(nlp=nlp, premises=test_texts1, hypothesis=test_texts2, num_unk=nr_unk,
                                              max_length=max_length)
        with open(processed_path + "test_x.pkl", "wb") as f:
            pickle.dump(test_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(path=configs[transformer_type] + "weights.pkl"):
        print(transformer_type, "weights matrix already extracted, now loading...")
        with open(configs[transformer_type] + "weights.pkl", "rb") as f:
            vectors = pickle.load(f)
    else:
        print(transformer_type, " weight matrix is not found, now extracting...")
        vectors = get_embeddings(vocab=nlp.vocab, nr_unk=nr_unk)
        with open(configs[transformer_type] + "weights.pkl", "wb") as f:
            pickle.dump(vectors, f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_x, train_labels, dev_x, dev_labels, test_x, test_labels, vectors

def decay_schedule(epoch, lr):
    if epoch == 0:
        print("epoch=", epoch, "and learning rate=", lr)
        return lr
    elif epoch != 0:
        lr = lr / 2
        print("epoch=", epoch, "and learning rate=", lr)
        return lr

def train_model(model_type, max_length, batch_size, nr_epoch, nr_hidden, nr_class, learning_rate,
                early_stopping, train_x, train_labels, dev_x, dev_labels, test_x, test_labels, vectors, result_path):
    """
    Model will be trained in this function. Currently it supports ESIM and Decomposable Attention models.
    :param model_type: type of the model and model save path. Either ESIM or Decomposable Attention.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param batch_size: represents the amount of data the model will train for each pass.
    :param nr_epoch: total number of times the model iterates trough all the training data.
    :param nr_hidden: hidden neuron size of the model.
    :param nr_class: number of classes that model will classify given pairs. Also the last layer of the model.
    :param learning_rate: learning rate parameter which will update the weights on back propagation.
    :param early_stopping: parameter that stops the training based on given condition.
    :param train_x: training data.
    :param train_labels: training labels.
    :param dev_x: developer data.
    :param dev_labels: developer labels.
    :param test_x: test data.
    :param test_labels: test data labels.
    :param vectors: embedding vectors of the tokens.
    :param result_path: path where accuracy and loss graphs will be saved along with the model history.
    :return: None
    """

    model = None

    if model_type == "esim":
        print("Architecture type =", model_type)
        model = esim_bilstm_model(vectors=vectors, max_length=max_length, nr_hidden=nr_hidden,
                                  nr_class=nr_class, learning_rate=learning_rate)

    elif model_type == "decomposable_attention":
        print("Architecture type =", model_type)
        model = decomposable_attention_model(vectors=vectors, max_length=max_length,
                                             nr_hidden=nr_hidden, nr_class=nr_class,
                                             learning_rate=learning_rate)
    # loss,accuracy,val_loss,val_accuracy
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                       patience=early_stopping, restore_best_weights=True)

    lr_scheduler = LearningRateScheduler(decay_schedule)

    ###############################
    # "This is for random sampling to test hyper-parameters."
    # idx = np.random.choice(np.arange(len(train_labels)), 25000, replace=False)
    # train_x = [train_x[0][idx], train_x[1][idx]]
    # train_labels = train_labels[idx]
    #
    # idx = np.random.choice(np.arange(len(dev_labels)), 1000, replace=False)
    # dev_x = [dev_x[0][idx], dev_x[1][idx]]
    # dev_labels = dev_labels[idx]

    ################################
    model.summary()

    history = model.fit(
        train_x,
        train_labels,
        validation_data=(dev_x, dev_labels),
        epochs=nr_epoch,
        batch_size=batch_size,
        verbose=1,
        callbacks=[es, lr_scheduler]
    )

    if not os.path.isdir(configs[model_type]):
        os.mkdir(configs[model_type])
    print("Saving to", configs[model_type])
    model.save(configs[model_type] + "model", save_format="h5")

    print('\n model history:', history.history)

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    with open(result_path + 'result_history.txt', 'w') as file:
        file.write(str(history.history))

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
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

    print("Evaluate on test data")
    results = model.evaluate(test_x, test_labels, batch_size=64)
    print("test loss, test acc:", results)

def main():
    train_x, train_labels, dev_x, dev_labels, test_x, test_labels, vectors = \
        spacy_word_transformer(processed_path=args.processed_path,
                               train_loc=args.train_loc,
                               dev_loc=args.dev_loc,
                               test_loc=args.test_loc,
                               max_length=args.max_length,
                               transformer_type=args.transformer_type,
                               nr_unk=args.nr_unk)

    train_model(model_type=args.model_type,
                max_length=args.max_length,
                batch_size=args.batch_size,
                nr_epoch=args.nr_epoch,
                nr_hidden=args.nr_hidden,
                nr_class=args.nr_class,
                learning_rate=args.learning_rate,
                early_stopping=args.early_stopping,
                train_x=train_x,
                train_labels=train_labels,
                dev_x=dev_x,
                dev_labels=dev_labels,
                test_x=test_x,
                test_labels=test_labels,
                vectors=vectors,
                result_path=args.result_path)


if __name__ == "__main__":
    plac.call(main)
