"""
This script works with Bert pretrained word weights extracted from bert model. It has the same idea that pretrained
word vectors use. Main difference is that bert uses full tokenizer which reduces the need of more tokens and vectors.
Instead, full tokenizer decomposes the word to its root and gives weight to root and suffix separately.
"""
import argparse

import numpy as np
import plac
import tensorflow as tf
from keras import Model
from keras.models import load_model

from bert_dependencies import tokenization
from utilities.utils import read_nli, attention_visualization, xml_test_file_reader, load_configurations, \
    predictions_to_html

configs = load_configurations()
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="demo",
                    help="This argument is to select whether to carry out 'evaluate' or 'demo' operation. Evaluate"
                         "operation takes labeled test data and measures the accuracy of the model. Demo operation"
                         "is for comparing unlabeled data. Demo support two individual sentences or list of sentences"
                         "as input data.")

parser.add_argument("--nli_type", type=str, default="snli mnli anli",
                    help="This parameter defines the train data which the model trained on. By specifying this"
                         "one can see the model behaviour based on trained data on prediction time. There are 4 main "
                         "nli dataset 'snli', 'mnli', 'anli', 'fewer'. One can combine each of these according to"
                         "their needs. Specify this by hand based on the model you will use on prediction time")

parser.add_argument("--transformer_type", type=str, default="bert",
                    help="Type of the transformer which will convert texts in to word-ids. This script is designed for"
                         "only bert. Parameter rakes only 'bert' option.")

parser.add_argument("--model_type", type=str, default="esim",
                    help="Type of the model that will be trained. "
                         "for ESIM model type 'esim' "
                         "for decomposable attention model type 'decomposable_attention'. ")

parser.add_argument("--visualization", type=bool, default=True,
                    help="shows attention heatmaps between two opinion sentences, best used with single"
                         "premise- hypothesis opinion sentences.")

parser.add_argument("--transformer_path", type=str, default=configs["transformer_paths"],
                    help="Main transformer model path which will convert the text in to word-ids and vectors. "
                         "transformer path has four sub paths.")

parser.add_argument("--max_length", type=str, default=configs["max_length"],
                    help="Max length of the sentences,longer sentences will be pruned and shorter ones will be zero"
                         "padded. Remember longer sentences means longer sequences to train. Select best length based"
                         "on your rig.")

parser.add_argument("--model_save_path", type=str, default=configs["model_paths"],
                    help="The path where trained NLI model is saved.")

parser.add_argument("--result_path", type=str, default=configs["results"],
                    help="path of the file where results and graphs will be saved.")

parser.add_argument("--test_loc", type=str, default=configs["nli_set_test"],
                    help="Test data location which will be used to measure the evaluation accuracy,")
args = parser.parse_args()

entailment_types = ["entailment", "contradiction", "neutral"]


def convert_examples_to_features(sentence, max_length, attention_heatmap, tokenizer):
    """
    Converts to unicode and tokenizes the examples. converts sentences in to features such as id's of the tokens,
     not full vector representations.
    :param sentence: opinion sentence.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param attention_heatmap: shows the attention heatmap of two opinion sentences.
    :param tokenizer: bert tokenizer object. loads the vocabulary from bert folder and performs full tokenizer.
    :return: word ids and tokens based on the visualization option.
    """

    line = tokenization.convert_to_unicode(sentence).strip()
    tokens = tokenizer.tokenize(line)

    if len(tokens) > max_length:
        tokens = tokens[0:max_length]

    input_ids_raw = tokenizer.convert_tokens_to_ids(tokens)

    while len(input_ids_raw) < max_length:
        input_ids_raw.append(0)
    assert len(input_ids_raw) == max_length

    word_ids = np.asarray(input_ids_raw).reshape((1, max_length))

    if attention_heatmap:
        return word_ids, tokens
    else:
        return word_ids


def evaluate(dev_loc, max_length, transformer_path, transformer_type, model_path, model_type):
    """
    This function is to measure model accuracy, it takes labeled test NLI data and performs model test on it.
    :param dev_loc: labeled test data location.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param transformer_path: path of the transformer  object.
    :param transformer_type: type of the transformer in this case it is 'bert'.
    :param model_path: path where the model is saved as h5 file.
    :param model_type: type of the model. either ESIM or Decomposable attention.
    :return: None
    """
    print("Loading trained NLI model")
    model = load_model(model_path[model_type] + "model.h5", custom_objects={"tf": tf})
    print("trained NLI model loaded")

    model.summary()
    model = Model(inputs=model.input,
                  outputs=[model.output, model.get_layer('sum_x1').output, model.get_layer('sum_x2').output])

    tokenizer = tokenization.FullTokenizer(
        vocab_file=transformer_path[transformer_type] + "vocab.txt", do_lower_case=True)

    premise, hypothesis, dev_labels = read_nli(dev_loc)

    total = 0.0
    true_p = 0.0

    for text1, text2, label in zip(premise, hypothesis, dev_labels):
        premise_features = convert_examples_to_features(sentence=text1, max_length=max_length,
                                                        attention_heatmap=False, tokenizer=tokenizer)

        hypothesis_features = convert_examples_to_features(sentence=text2, max_length=max_length,
                                                           attention_heatmap=False, tokenizer=tokenizer)

        outputs = model.predict([premise_features, hypothesis_features])
        # scores = outputs[0]
        if entailment_types[outputs[0].argmax()] == entailment_types[label.argmax()]:
            true_p += 1
        total += 1
    print("Entailment Model Accuracy is", true_p / total)


def demo(premise, hypothesis, transformer_path, transformer_type, model_path, model_type,
         max_length, attention_map, result_path, nli_type):
    """
    Performs demo operation using trained NLI model. Either takes two strings or list of strings and compares the
    premise - hypothesis pairwise and returns the NLI result.
    :param premise: opinion sentence
    :param hypothesis: opinion sentence
    :param transformer_path: path of the transformer object.
    :param transformer_type: type of the transformer in this case it is 'bert'.
    :param model_path: path where the model is saved as h5 file.
    :param model_type: type of the model. either ESIM or Decomposable attention.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param attention_map: boolean value to show attention heatmap of string based comparison.
    :param result_path: path of the file where the results will be saved.
    :param nli_type: type of the nli set which the model trained on.
    :return: None
    """
    print("Loading NLI model")
    model = load_model(model_path[model_type] + "model.h5", custom_objects={"tf": tf})
    print("NLI model loaded")

    model.summary()
    model = Model(inputs=model.input,
                  outputs=[model.output, model.get_layer('sum_x1').output, model.get_layer('sum_x2').output])

    tokenizer = tokenization.FullTokenizer(
        vocab_file=transformer_path[transformer_type] + "vocab.txt", do_lower_case=True)

    if type(premise) and type(hypothesis) is str:

        print("premise:", premise)
        print("hypothesis:", hypothesis)

        premise_features, premise_token = convert_examples_to_features(sentence=premise, max_length=max_length,
                                                                       attention_heatmap=True, tokenizer=tokenizer)

        hypothesis_features, hypothesis_token = convert_examples_to_features(sentence=hypothesis, max_length=max_length,
                                                                             attention_heatmap=True,
                                                                             tokenizer=tokenizer)

        outputs = model.predict([premise_features, hypothesis_features])
        scores = outputs[0]

        print("Entailment type is:", entailment_types[scores.argmax()], "\nEntailment confidence is: ", scores.max())
        if attention_map:
            attention_visualization(tokens1=premise_token, tokens2=hypothesis_token,
                                    attention1=outputs[1], attention2=outputs[2],
                                    results_path=result_path, transformer_type=transformer_type)

    elif type(premise) and type(hypothesis) is list:
        a = min(len(premise), len(hypothesis))
        premises = premise[:a]
        hypothesises = hypothesis[:a]

        prediction_type = []
        contradiction_score = []
        neutral_score = []
        entailment_score = []

        total = 0.0
        contradiction = 0.0
        entailment = 0.0
        neutral = 0.0

        for text1, text2 in zip(premises, hypothesises):
            premise_features = convert_examples_to_features(sentence=text1, max_length=max_length,
                                                            attention_heatmap=False, tokenizer=tokenizer)

            hypothesis_features = convert_examples_to_features(sentence=text2, max_length=max_length,
                                                               attention_heatmap=False, tokenizer=tokenizer)

            outputs = model.predict([premise_features, hypothesis_features])
            prediction = entailment_types[outputs[0].argmax()]

            prediction_type.append(prediction)
            contradiction_score.append(outputs[0][0][1])
            neutral_score.append(outputs[0][0][2])
            entailment_score.append(outputs[0][0][0])

            if prediction is 'contradiction':
                contradiction += 1
            elif prediction is 'entailment':
                entailment += 1
            elif prediction is 'neutral':
                neutral += 1
            total += 1

        print("total contradiction = ", contradiction / total)
        print("total entailment =", entailment / total)
        print("total neutral =", neutral / total)

        prediction_type.append(total)
        contradiction_score.append(contradiction)
        neutral_score.append(neutral)
        entailment_score.append(entailment)

        predictions_to_html(nli_type=nli_type,
                            premises=premises,
                            hypothesises=hypothesises,
                            prediction=prediction_type,
                            contradiction_score=contradiction_score,
                            neutral_score=neutral_score,
                            entailment_score=entailment_score,
                            result_path=result_path
                            )


def main():
    if args.mode == "evaluate":
        evaluate(dev_loc=args.test_loc,
                 max_length=args.max_length,
                 transformer_path=args.transformer_path,
                 transformer_type=args.transformer_type,
                 model_path=args.model_save_path,
                 model_type=args.model_type)

    elif args.mode == "demo":

        path = "/media/ulgen/Samsung/contradiction_data_depo/results/a/data/UKPConvArg1Strict-XML/"

        premise, _ = xml_test_file_reader(path=path + "christianity-or-atheism-_atheism.xml")
        hypothesis, _ = xml_test_file_reader(path=path + "christianity-or-atheism-_christianity.xml")

        demo(max_length=args.max_length,
             transformer_path=args.transformer_path,
             transformer_type=args.transformer_type,
             model_path=args.model_save_path,
             model_type=args.model_type,
             premise=premise,
             hypothesis=hypothesis,
             attention_map=args.visualization,
             result_path=args.result_path,
             nli_type=args.nli_type)


if __name__ == "__main__":
    plac.call(main)
