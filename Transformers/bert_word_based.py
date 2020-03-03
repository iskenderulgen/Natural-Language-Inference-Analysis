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
import os
import pickle
import random

import numpy as np
import tensorflow as tf

from Transformers.utils import read_snli
from bert import modeling, tokenization

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", type=int, default=64,
                    help="The maximum total input sequence length after WordPiece tokenization. "
                         "Sequences longer than this will be truncated, and sequences shorter "
                         "than this will be padded.")
parser.add_argument("--do_lower_case", type=bool, default=True,
                    help="Whether to lower case the input text. Should be True for uncased "
                         "models and False for cased models.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for predictions.")
parser.add_argument("--use_tpu", type=bool, default=False, help="Whether to use TPU or GPU/CPU.")
parser.add_argument("--master", default=None, help="If using a TPU, the address of the master.")
parser.add_argument("--num_tpu_cores", type=int, default=8,
                    help="Only used if `use_tpu` is True. Total number of TPU cores to use.")
parser.add_argument("--use_one_hot_embeddings", type=bool, default=False,
                    help="If True, tf.one_hot will be used for embedding lookups, otherwise "
                         "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
                         "since it is much faster.")
args = parser.parse_args()

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids":
                tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_type_ids":
                tf.constant(
                    all_input_type_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, mode):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % mode)

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        if use_tpu:

            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        all_layers = model.get_all_encoder_layers()

        predictions = {
            "unique_id": unique_ids,
        }

        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    # converts sentences in to features such as id's of the tokens, not full vector representations.
    total_sent_count = len(examples)
    sent_count = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            if len(tokens_a) > seq_length:
                tokens_a = tokens_a[0:seq_length]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        for token in tokens_a:
            tokens.append(token)

        # This is the part that tokens converted to ids.
        input_ids_raw = tokenizer.convert_tokens_to_ids(tokens)

        # (+100) is reserved for OOV tokens
        input_ids = []
        for i in range(len(input_ids_raw)):
            input_ids.append(100 + input_ids_raw[i])

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            # input_ids.append(0)
            random.randrange(100)
        assert len(input_ids) == seq_length

        # prints 5 example in to the console as an example
        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % example.unique_id)
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))

        features.append(input_ids)

        sent_count = sent_count + 1
        if sent_count % 50000 == 0:
            print("total sentence: " + str(sent_count) + " Total percent: " + str(
                sent_count / total_sent_count))

        # features stands for the token unique id's that represents vector.
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_sentences):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    total_sents = len(input_sentences)
    print(total_sents)

    if type(input_sentences) is np.ndarray or list:
        print("input file is array or list")
        for sentence in input_sentences:
            line = tokenization.convert_to_unicode(sentence).strip()
            examples.append(InputExample(unique_id=unique_id, text_a=line, text_b=None))
            unique_id += 1
        return examples, total_sents


def get_word_embedding_matrix(file_name, tensor_name, all_tensors,
                              all_tensor_names=False):
    embeds = []
    nr_unk = 100

    oov = np.random.normal(size=(100, 1024))
    oov = oov / oov.sum(axis=1, keepdims=True)

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

    embed_matrix = np.zeros((30522 + nr_unk, 1024), dtype="float32")
    embed_matrix[0: nr_unk, ] = oov
    embed_matrix[nr_unk:30522 + nr_unk, ] = np.asarray(embeds)

    return embed_matrix


def word_transformer(path, input_file):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=path + "bert/vocab.txt", do_lower_case=args.do_lower_case)

    examples, total_sent_count = read_examples(input_file)

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
        vectors_dev = word_transformer(path=path, input_file=dev_sentences)
        dev_x = [np.array(vectors_dev[: len(dev_texts1)]), np.array(vectors_dev[len(dev_texts2):])]
        with open(path + 'Processed_SNLI/Bert_Processed_WordLevel/dev_x.pkl', 'wb') as f:
            pickle.dump(dev_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(path + "/Processed_SNLI/Bert_Processed_WordLevel/weights.pkl"):
        print("embedding matrix is already exist")
        with open(path + 'Processed_SNLI/Bert_Processed_WordLevel/weights.pkl', 'rb') as f:
            word_weights = pickle.load(f)
    else:
        checkpoint_path = path + "bert/bert_model.ckpt"
        word_weights = get_word_embedding_matrix(file_name=checkpoint_path,
                                                 tensor_name='bert/embeddings/word_embeddings',
                                                 all_tensors=False, all_tensor_names=False)
        with open(path + "/Processed_SNLI/Bert_Processed_WordLevel/weights.pkl", 'wb') as f:
            pickle.dump(word_weights, f)

    print("bert feature extraction and embedding matrix extraction completed")

    return train_x, train_labels, dev_x, dev_labels, word_weights
