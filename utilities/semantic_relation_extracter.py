import json

import nltk
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization
from scipy import spatial

nltk.download('punkt')
from utilities.utils import load_configurations, set_memory_growth

set_memory_growth()
configs = load_configurations()


def read_nli(path):
    """
    Reads the NLI dataset from the given path
    :param path: path of the NLI dataset
    :return: returns the NLI data as list of texts.
    """
    texts1 = []
    texts2 = []
    labels = []
    with open(path, "r") as file_:
        for line in file_:
            nli_data = json.loads(line)
            label = nli_data["gold_label"]
            if label == "-":  # ignore '-'  SNLI entries
                continue
            texts1.append(nli_data["sentence1"])
            texts2.append(nli_data["sentence2"])
            labels.append(label)

    print("NLI dataset loaded")
    print(len(texts1))
    print(len(texts2))
    print(len(labels))
    return texts1, texts2, labels


def bert_encode(text, max_length, tokenizer):
    text = tokenizer.tokenize(text)

    text = text[:max_length - 2]
    input_sequence = ["[CLS]"] + text + ["[SEP]"]
    pad_len = max_length - len(input_sequence)

    tokens = tokenizer.convert_tokens_to_ids(input_sequence)
    tokens += [0] * pad_len
    pad_masks = [1] * len(input_sequence) + [0] * pad_len
    segment_ids = [0] * max_length

    output = dict(
        input_word_ids=tf.convert_to_tensor(np.asarray(tokens).reshape((1, 64)),
                                            name='input_word_ids', dtype=tf.int32),
        input_mask=tf.convert_to_tensor(np.asarray(pad_masks).reshape((1, 64)),
                                        name='input_mask', dtype=tf.int32),
        input_type_ids=tf.convert_to_tensor(np.asarray(segment_ids).reshape((1, 64)),
                                            name='input_type_ids', dtype=tf.int32)
    )

    return output


def compare(premise, hypothesis, label, bert, tokenizer):
    max_cosine = 0
    data = {}

    for line1 in nltk.sent_tokenize(premise):
        line1_bert_inputs = bert_encode(text=line1, max_length=64, tokenizer=tokenizer)
        line1_weights = bert(line1_bert_inputs)['sequence_output']

        for line2 in nltk.sent_tokenize(hypothesis):
            line2_bert_inputs = bert_encode(text=line2, max_length=64, tokenizer=tokenizer)
            line2_weights = bert(line2_bert_inputs)["sequence_output"]
            sim_score = 1 - spatial.distance.cosine(np.hstack(line1_weights.numpy()[0]),
                                                    np.hstack(line2_weights.numpy()[0]))

            if sim_score > max_cosine:
                max_cosine = sim_score
                data['sentence1'] = line1
                data['sentence2'] = line2

    data['gold_label'] = label

    with open("/home/ulgen/Downloads/split_results/splittled_nli.jsonl", "a") as outfile:
        outfile.write(json.dumps(data) + "\n")


def main():
    bert_tf_path = "/media/ulgen/Samsung/contradiction_data/transformers/tf_hub"

    bert_encoder = hub.KerasLayer(handle=bert_tf_path, trainable=False)
    vocab_file = bert_encoder.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_encoder.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    text1, text2, label = read_nli("/media/ulgen/Samsung/contradiction_data/nli_sets/train.jsonl")

    count = 0

    for premise, hypothesis, label in zip(text1, text2, label):
        compare(premise=premise, hypothesis=hypothesis, label=label, bert=bert_encoder, tokenizer=tokenizer)

        count = count + 1
        if count % 5000 == 0:
            print("processed Sentence:", str(count),
                  "Processed Percentage:", str(round(count / len(text1), 4) * 100))


main()

# vocab_file = "/media/ulgen/Samsung/contradiction_data/transformers/bert_tf_hub/assets/vocab.txt"
