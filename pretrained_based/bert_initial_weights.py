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
This code coverts premises and hypothesis using bert_dependencies-initial-weights. Process is straight-forwards similar
to pre-trained approaches. Instead of prediction based approach we extract initial-word-matrix from bert_dependencies-model
and use this to create id's of sentences. Thanks to bert_dependencies's full-tokenizer we can achieve good results with
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
import tensorflow as tf

from utils.utils import read_snli
from bert_dependencies import tokenization

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", type=int, default=64,
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


def convert_examples_to_features(path, premises, hypothesis, seq_length):
    tf.logging.set_verbosity(tf.logging.INFO)

    sents = premises + hypothesis
    total_sent_count = len(sents)
    processed_sent_count = 0
    features = []
    print("Total sentences to be processed: ", total_sent_count)

    start_time = datetime.datetime.now()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=path + "transformers/bert/vocab.txt", do_lower_case=args.do_lower_case)

    for (ex_index, example) in enumerate(sents):
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
            print("Processed sentence: " + str(processed_sent_count) + " Processed percent: " +
                  str(round(processed_sent_count / total_sent_count, 4) * 100))

    finish_time = datetime.datetime.now()
    total_time = finish_time - start_time
    print("Total time spent to create token ID's of sentences: ", total_time)

    return [np.array(features[: len(premises)]), np.array(features[len(premises):])]


def extract_initial_word_embedding_matrix(file_name, tensor_name, all_tensors, all_tensor_names=False):
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
    embed_matrix = np.zeros((bert_vector_size, 1024), dtype="float32")
    embed_matrix[0:bert_vector_size] = np.asarray(embeds)

    return embed_matrix


def bert_initial_weights_transformer(path, train_loc, dev_loc, transformer_type):
    print("Starting to pre-process using bert-initial-word_embeddings. Transformer type is ", transformer_type)

    train_texts1, train_texts2, train_labels = read_snli(train_loc)
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)

    if not os.path.isdir(path + "Processed_SNLI"):
        print("Processed_SNLI directory is not exist, now created")
        os.mkdir(path + "Processed_SNLI")

    if os.path.isfile(path=path + "Processed_SNLI/"+transformer_type+"/train_x.pkl"):
        print("Pre-Processed train file is found now loading")
        with open(path + "Processed_SNLI/"+transformer_type+"/train_x.pkl", "rb") as f:
            train_x = pickle.load(f)
    else:
        print("There is no pre-processed file of train_X, Pre-Process will start now")

        train_x = convert_examples_to_features(path=path, premises=train_texts1, hypothesis=train_texts2,
                                               seq_length=args.max_seq_length)
        with open(path + "Processed_SNLI/"+transformer_type+"/train_x.pkl", "wb") as f:
            pickle.dump(train_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(path=path + "Processed_SNLI/"+transformer_type+"/dev_x.pkl"):
        print("Pre-Processed dev file is found now loading")
        with open(path + "Processed_SNLI/"+transformer_type+"/dev_x.pkl", "rb") as f:
            dev_x = pickle.load(f)
    else:
        print("There is no pre-processed file of dev_X, Pre-Process will start now")
        dev_x = convert_examples_to_features(path=path, premises=dev_texts1, hypothesis=dev_texts2,
                                             seq_length=args.max_seq_length)
        with open(path + "Processed_SNLI/"+transformer_type+"/dev_x.pkl", "wb") as f:
            pickle.dump(dev_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(path + "/Processed_SNLI/"+transformer_type+"/weights.pkl"):
        print("embedding matrix is already exist")
        with open(path + "Processed_SNLI/"+transformer_type+"/weights.pkl", "rb") as f:
            word_weights = pickle.load(f)
    else:
        checkpoint_path = path + "transformers/bert/bert_model.ckpt"
        word_weights = extract_initial_word_embedding_matrix(file_name=checkpoint_path,
                                                             tensor_name='bert/embeddings/word_embeddings',
                                                             all_tensors=False, all_tensor_names=False)
        with open(path + "/Processed_SNLI/"+transformer_type+"/weights.pkl", 'wb') as f:
            pickle.dump(word_weights, f)

    print("bert initial-word-weights based feature extraction and embedding matrix extraction completed")

    return train_x, train_labels, dev_x, dev_labels, word_weights
