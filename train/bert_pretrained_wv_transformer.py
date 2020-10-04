# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code coverts premises and hypothesis using pre trained initial bert word weights. Process is straight forward and
similar to pre-trained approaches. Instead of prediction based approach we extract initial-word-matrix from bert model
and use this to create id's of sentences. Thanks to bert's full-tokenizer we can achieve good results with
35,522 token & vector.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import pickle
import numpy as np
import plac
import tensorflow as tf
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from models.decomposable_attention import decomposable_attention_model
from models.esim import esim_bilstm_model
from utils.utils import read_snli, load_configurations
from bert_dependencies import tokenization

configs = load_configurations()

parser = argparse.ArgumentParser()
parser.add_argument("--transformer_type", type=str, default="bert",
                    help="type of the transformer which will convert texts in to word-ids. This script only for bert,"
                         "so default transformer type remains 'bert'")

parser.add_argument("--embedding_type", type=str, default="word",
                    help="For word embedding base models use 'word' keyword,"
                         "For sentence embedding base models use 'sentence' keyword. "
                         "Required embedding layer will be triggered based on selection")

parser.add_argument("--model_type", type=str, default="esim",
                    help="Type of the model that will be trained. "
                         "for ESIM model type 'esim' "
                         "for decomposable attention model type 'decomposable_attention'. ")

parser.add_argument("--transformer_path", type=str, default=configs["transformer_paths"],
                    help="main transformer model path which will convert the text in to word-ids and vectors. "
                         "transformer path has four sub paths, load_nlp module will carry out the sub model paths"
                         "based on transformer type selection")

parser.add_argument("--train_loc", type=str, default=configs["nli_set_train"],
                    help="Train data location which will be processed via transformers and be saved to processed_path "
                         "location")

parser.add_argument("--dev_loc", type=str, default=configs["nli_set_dev"],
                    help="Train data dev location which will be used to measure train accuracy while training model,"
                         "files will be processed using transformer and be saved to processed path")

parser.add_argument("--max_length", type=str, default=configs["max_length"],
                    help="max length of the sentences,longer sentences will be pruned and shorter ones will be zero"
                         "padded. Remember longer sentences means longer sequences to train. Select best length based"
                         "on your rig.")

parser.add_argument("--nr_unk", type=int, default=configs["nr_unk"],
                    help="number of unknown vectors which will be used for padding the short sentences to desired"
                         "length.Nr unknown vectors will be created using random module")

parser.add_argument("--processed_path", type=str, default=configs["processed_nli"],
                    help="Path where the transformed texts will be saved to as word-ids. Will be used for embedding"
                         "layer of the train models.")

parser.add_argument("--model_save_path", type=str, default=configs["model_paths"],
                    help="The path where trained NLI model will be saved.")

parser.add_argument("--batch_size", type=int, default=configs["batch_size"],
                    help="batch size of model, it represents the amount of data the model will train for each pass.")

parser.add_argument("--nr_epoch", type=int, default=configs["nr_epoch"],
                    help="Total number of times that model will iterate trough the data.")

parser.add_argument("--nr_hidden", type=int, default=configs["nr_hidden"],
                    help="hidden neuron size of the model")

parser.add_argument("--nr_class", type=int, default=configs["nr_class"],
                    help="number of class that will model classify the data into. Also represents the last layer of"
                         "the model.")

parser.add_argument("--learning_rate", type=float, default=configs["learn_rate"],
                    help="learning rate parameter that represent the constant which will be multiplied with the data"
                         "in each back propagation")

parser.add_argument("--result_path", type=str, default=configs["results"],
                    help="path of the file where trained model loss and accuracy graphs will be saved.")
args = parser.parse_args()


def convert_examples_to_features(premises, hypothesis, seq_length, bert_directory):
    """
    This function used bert transformer to create word ids of tokens. This process is similar to pre trained word
    weight transformation, uses the bert initial pretrained word weights to create vectors. Compared to priors bert
    pretrained initial word weights are much smaller than old pretrained vectors thanks to bert's full word tokenizer.
    :param premises: opinion sentence
    :param hypothesis: opinion sentence
    :param seq_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param bert_directory: path of the transformer object. Bert files location.
    :return: returns word ids as a list.
    """

    tf.logging.set_verbosity(tf.logging.INFO)
    sentences = premises + hypothesis
    total_sent_count = len(sentences)
    print("Total sentences to be processed: ", total_sent_count)
    processed_sent_count = 0
    features = []
    start_time = datetime.datetime.now()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=bert_directory + "vocab.txt", do_lower_case=args.do_lower_case)

    for (ex_index, example) in enumerate(sentences):
        line = tokenization.convert_to_unicode(example).strip()
        tokens = tokenizer.tokenize(line)

        if len(tokens) > seq_length:
            tokens = tokens[0:seq_length]

        input_ids_raw = tokenizer.convert_tokens_to_ids(tokens)

        while len(input_ids_raw) < seq_length:
            input_ids_raw.append(0)
        assert len(input_ids_raw) == seq_length

        features.append(input_ids_raw)

        processed_sent_count = processed_sent_count + 1
        if processed_sent_count % 50000 == 0:
            print("Processed sentence: ", str(processed_sent_count),
                  "Processed percent: ", str(round(processed_sent_count / total_sent_count, 4) * 100))

    finish_time = datetime.datetime.now()
    print("Total time spent to create token ID's of sentences: ", finish_time - start_time)

    return [np.array(features[: len(premises)]), np.array(features[len(premises):])]


def extract_initial_word_embedding_matrix(file_name, tensor_name, all_tensors, all_tensor_names=False):
    """
    This function exports the bert initial pre trained word weights from the tensor model. This weights will be used
    to create word-ids - weights matrix to be used in embedding layer. Currently bert contains 35.552 words and weights.
    :param file_name: bert pretrained model file name
    :param tensor_name: name of the tensor which will be extracted
    :param all_tensors: whether to print all tensors or not.
    :param all_tensor_names: whether to print all tensors or not.
    :return: returns word weights that extracted from bert model.
    """
    embeds = []

    reader = tf.pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors or all_tensor_names:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            print("tensor_name: ", key)
            if all_tensors:
                print(reader.get_tensor(key))
    elif not tensor_name:
        print(reader.debug_string().decode("utf-8"))
    else:
        print("tensor_name: ", tensor_name)
        # print(reader.get_tensor(tensor_name))
        embeds.append(reader.get_tensor(tensor_name))

    bert_vector_size = np.asarray(embeds).shape[1]
    embedding_matrix = np.zeros((bert_vector_size, 1024), dtype="float32")
    embedding_matrix[0:bert_vector_size] = np.asarray(embeds)

    return embedding_matrix


def bert_pretrained_transformer(transformer_path, transformer_type, train_loc, dev_loc,
                                max_length, processed_path):
    """
    This function reads NLI sets and processes them trough the functions above. Takes sentences as list and transforms
    them in to word-id matrix. This word_id matrix will be then saved to disk as pkl file to be read and used in
    embedding layer of the madel.
    :param transformer_path: path of the transformer bert object.
    :param transformer_type: type of the transformer. This parameter only takes 'bert' as transformer.
    :param train_loc: training NLI jsonl date location.
    :param dev_loc: dev NLI jsonl date location
    :param max_length: max length of the sentence. Longer ones will be pruned shorter ones will be padded.
    :param processed_path: path where the processed files will be based.
    :return: returns train - dev set and corresponding labels with word weights.
    """
    print("starting to pre-process using bert-initial word embeddings.")
    train_texts1, train_texts2, train_labels = read_snli(train_loc)
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)

    if not os.path.isdir(processed_path):
        print("Processed_SNLI directory is not exist, now created")
        os.mkdir(processed_path)

    if os.path.isfile(path=processed_path + "train_x.pkl"):
        print("Pre-Processed train file is found now loading")
        with open(processed_path + "train_x.pkl", "rb") as f:
            train_x = pickle.load(f)
    else:
        print("There is no pre-processed file of train_X, Pre-Process will start now")

        train_x = convert_examples_to_features(premises=train_texts1, hypothesis=train_texts2, seq_length=max_length,
                                               bert_directory=transformer_path[transformer_type])
        with open(processed_path + "train_x.pkl", "wb") as f:
            pickle.dump(train_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(path=processed_path + "dev_x.pkl"):
        print("Pre-Processed dev file is found now loading")
        with open(processed_path + "dev_x.pkl", "rb") as f:
            dev_x = pickle.load(f)
    else:
        print("There is no pre-processed file of dev_X, Pre-Process will start now")
        dev_x = convert_examples_to_features(premises=dev_texts1, hypothesis=dev_texts2, seq_length=max_length,
                                            bert_directory=transformer_path[transformer_type])
        with open(processed_path + "dev_x.pkl", "wb") as f:
            pickle.dump(dev_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(transformer_path[transformer_type] + "weights.pkl"):
        print("Embedding matrix is already exist")
        with open(transformer_path[transformer_type] + "weights.pkl", "rb") as f:
            word_weights = pickle.load(f)
    else:
        checkpoint_path = transformer_path[transformer_type] + "bert_model.ckpt"
        word_weights = extract_initial_word_embedding_matrix(file_name=checkpoint_path,
                                                             tensor_name='bert/embeddings/word_embeddings',
                                                             all_tensors=False, all_tensor_names=False)
        with open(transformer_path[transformer_type] + "weights.pkl", 'wb') as f:
            pickle.dump(word_weights, f)

    print("Bert initial word weights based feature extraction and embedding matrix extraction completed.")

    return train_x, train_labels, dev_x, dev_labels, word_weights


def train_model(model_save_path, model_type, max_length, batch_size, nr_epoch,
                nr_hidden, nr_class, learning_rate, embedding_type, early_stopping,
                train_x, train_labels, dev_x, dev_labels, vectors, result_path):
    """
    Model will be trained in this function. Currently it supports ESIM and Decomposable Attention models.
    :param model_save_path: path where the model will be saved as h5 file.
    :param model_type: type of the model. either ESIM or Decomposable attention.
    :param max_length: max length of the sentence / sequence.
    :param batch_size: Size of the train data will be feed forwarded on each iteration.
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

    model.summary()

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

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(result_path + 'accuracy.png', bbox_inches='tight')

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(result_path + 'loss.png', bbox_inches='tight')

    print('\n model history:', history.history)

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    with open(result_path + 'result_history.txt', 'w') as file:
        file.write(str(history.history))

    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)
    print("Saving to", model_save_path)

    model.save(model_save_path[model_type] + "model.h5")


def main():
    train_x, train_labels, dev_x, dev_labels, vectors = \
        bert_pretrained_transformer(transformer_path=args.transformer_path,
                                    transformer_type=args.transformer_type,
                                    train_loc=args.train_loc,
                                    dev_loc=args.dev_loc,
                                    max_length=args.max_length,
                                    processed_path=args.processed_path)

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
