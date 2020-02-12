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

import codecs
import collections
import json
import pickle
import re
import numpy as np
import os
import pprint

from bert import modeling, tokenization
import tensorflow as tf

path = "/home/ulgen/Documents/Python_Projects/Contradiction/data/bert/"
id_save_path = '/home/ulgen/Documents/Python_Projects/Contradiction/data/Processed_SNLI/Bert_Processed_WordLevel/'

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "max_seq_length", 64,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


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

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
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
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

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
            # Account for [CLS] and [SEP] with "- 2"
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

        # bizim icin önemli olan kısım bu input_ids kısmı.
        input_ids_raw = tokenizer.convert_tokens_to_ids(tokens)

        input_ids = []
        for i in range(len(input_ids_raw)):
            input_ids.append(100 + input_ids_raw[i])

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)

        assert len(input_ids) == seq_length

        # prints 5 example in to the console as an example
        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (example.unique_id))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))

        features.append(input_ids)

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
            line = tokenization.convert_to_unicode(sentence)
            examples.append(InputExample(unique_id=unique_id, text_a=line, text_b=None))
            unique_id += 1
        return examples, total_sents


def get_word_embedding_matrix(file_name, tensor_name, all_tensors,
                              all_tensor_names=False,
                              count_exclude_pattern=""):
    embeds = []
    nr_unk = 100
    """Prints tensors in a checkpoint file.

    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.

    If `tensor_name` is provided, prints the content of the tensor.

    Args:
      file_name: Name of the checkpoint file.
      tensor_name: Name of the tensor in the checkpoint file to print.
      all_tensors: Boolean indicating whether to print all tensors.
      all_tensor_names: Boolean indicating whether to print all tensor names.
      count_exclude_pattern: Regex string, pattern to exclude tensors when count.
    """
    oov = np.random.normal(size=(100, 768))
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

    vectors = np.zeros((30522 + nr_unk, 768), dtype="float32")
    vectors[0: nr_unk, ] = oov
    vectors[nr_unk:30522 + nr_unk, ] = np.asarray(embeds)

    return vectors


def word_transformer(bert_directory, input_file):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=bert_directory + "vocab.txt", do_lower_case=FLAGS.do_lower_case)

    examples, total_sent_count = read_examples(input_file)

    # This is the data which we'll need to export for word based level.
    features = convert_examples_to_features(
        examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    checkpoint_path = path + "bert_model.ckpt"

    if os.path.isfile(id_save_path + "weights.pkl"):
        print("weights are exist")
    else:
        word_weights = get_word_embedding_matrix(file_name=checkpoint_path,
                                                 tensor_name='bert/embeddings/word_embeddings',
                                                 all_tensors=False, all_tensor_names=False)
        with open(id_save_path + "weights.pkl", 'wb') as f:
            pickle.dump(word_weights, f)

    print("its finished")

    return features


def bert_word_based_transformer():
    print("Processing texts using bert")
    if os.path.isfile(path=path + "Processed_SNLI/Bert_Processed_WordLevel/train_x.pkl"):
        print("Pre-Processed train file is found now loading")
        with open(path + 'Processed_SNLI/Bert_Processed_WordLevel/train_x.pkl', 'rb') as f:
            train_x = pickle.load(f)
    else:
        print("There is no pre-processed file of train_X, Pre-Process will start now")
        train_sentences = train_texts1 + train_texts2
        train_ids = word_transformer(bert_directory=bert_dir, input_file=train_sentences)
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
        vectors_dev = word_transformer(bert_directory=bert_dir, input_file=dev_sentences)
        dev_x = [np.array(vectors_dev[: len(dev_texts1)]), np.array(vectors_dev[len(dev_texts2):])]
        with open(path + 'Processed_SNLI/Bert_Processed_WordLevel/dev_x.pkl', 'wb') as f:
            pickle.dump(dev_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_x, train_labels, dev_x, dev_labels