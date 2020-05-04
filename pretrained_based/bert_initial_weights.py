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
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import pickle

import numpy as np
import tensorflow as tf

from pretrained_based.utils import read_snli
from bert import tokenization

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", type=int, default=50,
                    help="The maximum total input sequence length after WordPiece tokenization. "
                         "Sequences longer than this will be truncated, and sequences shorter "
                         "than this will be padded.")
parser.add_argument("--do_lower_case", type=bool, default=True,
                    help="Whether to lower case the input text. Should be True for uncased "
                         "models and False for cased models.")
parser.add_argument("--use_one_hot_embeddings", type=bool, default=False,
                    help="If True, tf.one_hot will be used for embedding lookups, otherwise "
                         "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
                         "since it is much faster.")
args = parser.parse_args()


def convert_examples_to_features(examples, seq_length, tokenizer):
    total_sent_count = len(examples)
    sent_count = 0
    features = []

    start_time = datetime.datetime.now()

    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example)

        if len(tokens) > seq_length:
            tokens = tokens[0:seq_length]

        # This is the part that tokens converted to ids.
        input_ids_raw = tokenizer.convert_tokens_to_ids(tokens)

        # pad up to the sequence length.
        while len(input_ids_raw) < seq_length:
            input_ids_raw.append(0)
        assert len(input_ids_raw) == seq_length

        features.append(input_ids_raw)

        sent_count = sent_count + 1
        if sent_count % 50000 == 0:
            print("Processed sentence: " + str(sent_count) + " Processed percent: " +
                  str(round(sent_count / total_sent_count, 4) * 100))

    finish_time = datetime.datetime.now()
    total_time = finish_time - start_time
    print("Total time spent to create token ID's of sentences: ", total_time)

    return features


def read_examples(input_sentences):
    """Read a list of `InputExample`s from an input file."""

    if type(input_sentences) is np.ndarray or list:
        print("input file is array or list")

        examples = []
        total_sentences = len(input_sentences)
        print("Total sentences to be read: ", total_sentences)

        for sentence in input_sentences:
            line = tokenization.convert_to_unicode(sentence).strip()
            examples.append(line)
        return examples

    else:
        return TypeError


def extract_initial_word_embedding_matrix(file_name, tensor_name, all_tensors, all_tensor_names=False):
    embeds = []
    bert_vector_size = 30522

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

    embed_matrix = np.zeros((bert_vector_size, 1024), dtype="float32")
    embed_matrix[0:bert_vector_size] = np.asarray(embeds)
    return embed_matrix


def word_transformer(path, input_file):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=path + "bert/vocab.txt", do_lower_case=args.do_lower_case)

    examples = read_examples(input_file)

    # This is the data which we'll need to export for word based level.
    features = convert_examples_to_features(
        examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)

    return features


def bert_word_based_transformer(path, train_loc, dev_loc, transformer_type):
    print("Transformer type is ", transformer_type)

    train_texts1, train_texts2, train_labels = read_snli(train_loc)
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)

    if os.path.isfile(path=path + "Processed_SNLI/Bert_Processed_WordLevel/train_x.pkl"):
        print("Pre-Processed train file is found now loading")
        with open(path + 'Processed_SNLI/Bert_Processed_WordLevel/train_x.pkl', 'rb') as f:
            train_x = pickle.load(f)
    else:
        print("There is no pre-processed file of train_X, Pre-Process will start now")
        train_sentences = train_texts1 + train_texts2
        train_ids = word_transformer(path=path, input_file=train_sentences)
        train_x = [np.array(train_ids[: len(train_texts1)]), np.array(train_ids[len(train_texts2):])]
        with open(path + 'Processed_SNLI/Bert_Processed_WordLevel/train_x.pkl', 'wb') as f:
            pickle.dump(train_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(path=path + "Processed_SNLI/Bert_Processed_WordLevel/dev_x.pkl"):
        print("Pre-Processed dev file is found now loading")
        with open(path + 'Processed_SNLI/Bert_Processed_WordLevel/dev_x.pkl', 'rb') as f:
            dev_x = pickle.load(f)
    else:
        print("There is no pre-processed file of dev_X, Pre-Process will start now")
        dev_sentences = dev_texts1 + dev_texts2
        dev_ids = word_transformer(path=path, input_file=dev_sentences)
        dev_x = [np.array(dev_ids[: len(dev_texts1)]), np.array(dev_ids[len(dev_texts2):])]
        with open(path + 'Processed_SNLI/Bert_Processed_WordLevel/dev_x.pkl', 'wb') as f:
            pickle.dump(dev_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(path + "/Processed_SNLI/Bert_Processed_WordLevel/weights.pkl"):
        print("embedding matrix is already exist")
        with open(path + 'Processed_SNLI/Bert_Processed_WordLevel/weights.pkl', 'rb') as f:
            word_weights = pickle.load(f)
    else:
        checkpoint_path = path + "bert/bert_model.ckpt"
        word_weights = extract_initial_word_embedding_matrix(file_name=checkpoint_path,
                                                             tensor_name='bert/embeddings/word_embeddings',
                                                             all_tensors=False, all_tensor_names=False)
        with open(path + "/Processed_SNLI/Bert_Processed_WordLevel/weights.pkl", 'wb') as f:
            pickle.dump(word_weights, f)

    print("bert feature extraction and embedding matrix extraction completed")

    return train_x, train_labels, dev_x, dev_labels, word_weights
