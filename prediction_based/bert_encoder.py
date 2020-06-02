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

import numpy as np
import tensorflow as tf

from bert_dependencies import modeling, tokenization
from utils.utils import read_snli

parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=str, default="-1", help="Choose the layers that will be extracted")
parser.add_argument("--max_seq_length", type=int, default=50,
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
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

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
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (example.unique_id))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
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
    print("Total sentences to be processed is:", total_sents)

    if type(input_sentences) is np.ndarray or list:
        print("input file is array or list")
        for sentence in input_sentences:
            line = tokenization.convert_to_unicode(sentence).strip()
            examples.append(InputExample(unique_id=unique_id, text_a=line, text_b=None))
            unique_id += 1
        return examples, total_sents


def sentence_transformer(bert_directory, premises, hypothesis, feature_type):
    tf.logging.set_verbosity(tf.logging.INFO)

    sents = premises + hypothesis
    layer_indexes = [int(x) for x in args.layers.split(",")]

    bert_config = modeling.BertConfig.from_json_file(bert_directory + "bert_config.json")

    tokenizer = tokenization.FullTokenizer(
        vocab_file=bert_directory + "vocab.txt", do_lower_case=args.do_lower_case)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=args.master,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=args.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    examples, total_sent_count = read_examples(sents)

    # This is the data which we'll need to export for word based level.
    features = convert_examples_to_features(
        examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=bert_directory + "bert_model.ckpt",
        layer_indexes=layer_indexes,
        use_tpu=args.use_tpu,
        use_one_hot_embeddings=args.use_one_hot_embeddings)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=args.use_tpu,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=args.batch_size)

    # max seq length can be imported as parameter
    input_fn = input_fn_builder(
        features=features, seq_length=args.max_seq_length)

    processed_sent_count = 0
    sentence_vectors = []

    for result in estimator.predict(input_fn, yield_single_examples=True):
        unique_id = int(result["unique_id"])
        feature = unique_id_to_feature[unique_id]
        for (i, token) in enumerate(feature.tokens):
            for (j, layer_index) in enumerate(layer_indexes):
                layer_output = result["layer_output_%d" % j]
                layers_output_flat = [round(float(x), 6) for x in layer_output[i:(i + 1)].flat]
                sentence_vectors.append(layers_output_flat)
            # print("Token budur = kontrolden sonra sil bu satırı = ", token)
            break

        processed_sent_count = processed_sent_count + 1
        if processed_sent_count % 5000 == 0:
            print("processed Sentence: " + str(processed_sent_count) + " Processed Percentage: " +
                  str(round(processed_sent_count / len(sents), 4) * 100))

    print("Pre Processing using Bert prediction based sentence encoder is finished. Vectors are now being saved")
    return [np.array(sentence_vectors[: len(premises)]), np.array(sentence_vectors[len(premises):])]


def bert_transformer(path, train_loc, dev_loc, feature_type):
    print("Pre - Processing sentences using prediction based bert_dependencies sentence approach")

    train_texts1, train_texts2, train_labels = read_snli(train_loc)
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)

    if not os.path.isdir(path + "Processed_SNLI"):
        print("Processed_SNLI directory is not exist, now created")
        os.mkdir(path + "Processed_SNLI")

    if os.path.isfile(path=path + "Processed_SNLI/" + feature_type + "/dev_x.pkl"):
        print("Pre-Processed dev file is found now loading")
        with open(path + "Processed_SNLI/" + feature_type + "/dev_x.pkl", "rb") as f:
            dev_x = pickle.load(f)
    else:
        print("There is no pre-processed file of dev_X, Pre-Process will start now")
        dev_x = sentence_transformer(bert_directory=path + "transformers/bert/", premises=dev_texts1,
                                     hypothesis=dev_texts2, feature_type=feature_type)
        with open(path + "Processed_SNLI/" + feature_type + "/dev_x.pkl", "wb") as f:
            pickle.dump(dev_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(path=path + "Processed_SNLI/" + feature_type + "/train_x.pkl"):
        print("Pre-Processed train file is found now loading")
        with open(path + "Processed_SNLI/" + feature_type + "/train_x.pkl", "rb") as f:
            train_x = pickle.load(f)
    else:
        print("There is no pre-processed file of train_X, Pre-Process will start now")
        train_x = sentence_transformer(bert_directory=path + "transformers/bert/", premises=train_texts1,
                                       hypothesis=train_texts2, feature_type=feature_type)
        with open(path + "Processed_SNLI/" + feature_type + "/train_x.pkl", "wb") as f:
            pickle.dump(train_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_x, train_labels, dev_x, dev_labels
