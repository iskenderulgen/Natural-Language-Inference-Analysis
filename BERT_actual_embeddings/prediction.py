"""
This script works with Bert pretrained word weights extracted from bert model. It has the same idea that pretrained
word vectors use. Main difference is that bert uses full tokenizer which reduces the need of more tokens and vectors.
Instead, full tokenizer decomposes the word to its root and gives weight to root and suffix separately.
"""
import argparse

import numpy as np
import plac
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

from bert import tokenization
from utilities.utils import read_nli, attention_visualization, load_configurations, \
    predictions_to_html, set_memory_growth

set_memory_growth()
configs = load_configurations()
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="evaluate",
                    help="This argument is to select whether to carry out 'evaluate' or 'demo' operation. Evaluate"
                         "operation takes labeled test data and measures the accuracy of the model. Demo operation"
                         "is for real-life usage. Demo compares two individual sentence or list of sentences"
                         "as input data.")

parser.add_argument("--nli_type", type=str, default="snli",
                    help="This parameter defines the train data which the model trained with. By specifying this"
                         "one can see the model behaviour on prediction time based on train data. There are 3 main "
                         "nli dataset 'snli', 'mnli', 'anli'. One can combine each of these according to their needs."
                         "If you combine train sets, dont use underline to define combination. Send parameter with one"
                         "blank space. It will shorten the html cell size. For example 'snli mnli' for combination of "
                         "snli and mnli train sets. This will be used for result columns and graphs.")

parser.add_argument("--transformer_type", type=str, default="bert_embeddings",
                    help="Type of the transformer which will convert texts in to word-ids. Also carries the path "
                         "information of transformer object. This script is designed for only bert actual embeddings."
                         "Parameter takes only 'bert_embeddings' option.")

parser.add_argument("--model_type", type=str, default="esim",
                    help="Type of the model architecture that the model is trained on. This parameter also carries the "
                         "model save path information hence this is used for both defining architecture and carrying "
                         "path information."
                         "for ESIM model use 'esim' "
                         "for Decomposable Attention model use 'decomposable_attention'.")

parser.add_argument("--visualization", type=bool, default=True,
                    help="shows attention heatmaps between two opinion sentences, best used with single"
                         "premise- hypothesis opinion sentence.")

parser.add_argument("--max_length", type=str, default=configs["max_length"],
                    help="Max length of the sentences, longer sentences will be pruned and shorter ones will be zero"
                         "padded. longer sentences mean longer sequences to train. Pick best length based on your rig.")

parser.add_argument("--test_loc", type=str, default=configs["nli_set_test"],
                    help="Test data location which will be used to measure the accuracy of the model")

parser.add_argument("--result_path", type=str, default=configs["results"],
                    help="path of the folder where results and graphs will be saved.")
args = parser.parse_args()

entailment_types = ["entailment", "contradiction", "neutral"]


def convert_examples_to_features(text, max_length, tokenizer, attention_heatmap):
    """
    Converts the given pairs to unicode and tokenizes the examples to map into features such as ids of tokens. IDs of
    tokens are mapped using the lookup table extracted from BERT vector - vocabulary pairs. This does not  provide
    full vector representations.
    :param text: opinion sentence.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param tokenizer: bert tokenizer object. loads the vocabulary from bert folder and performs full tokenizer.
    :param attention_heatmap: boolean value to show attention heatmap of premise - hypothesis comparison.
    :return: word ids and tokens based on the visualization option.
    """

    tokens = tokenizer.tokenize(text)

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


def evaluate(test_loc, max_length, transformer_type, model_type):
    """
    Evaluates the trained NLI model with labeled NLI test data and prints accuracy metric.
    :param test_loc: labeled evaluation test data location.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param transformer_type: type of the transformer in this case it is 'bert_embeddings'.
    :param model_type: trained model architecture type. It is determined with argparse in the argument section.
    :return: None
    """
    print("Loading trained NLI model")
    model = load_model(configs[model_type] + "model")
    print("trained NLI model loaded")

    model.summary()
    model = Model(inputs=model.input, outputs=model.output)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=configs[transformer_type] + "vocab.txt", do_lower_case=True)

    premise, hypothesis, dev_labels = read_nli(test_loc)

    total = 0.0
    true_p = 0.0

    for text1, text2, label in zip(premise, hypothesis, dev_labels):
        premise_features = convert_examples_to_features(text=text1, max_length=max_length,
                                                        tokenizer=tokenizer, attention_heatmap=False)

        hypothesis_features = convert_examples_to_features(text=text2, max_length=max_length,
                                                           tokenizer=tokenizer, attention_heatmap=False)

        outputs = model.predict([premise_features, hypothesis_features])
        # scores = outputs[0]
        if entailment_types[outputs[0].argmax()] == entailment_types[label.argmax()]:
            true_p += 1
        total += 1
    print("NLI Model Accuracy is:", true_p / total)


def demo(premise, hypothesis, transformer_type, nli_type, model_type, max_length, attention_map, result_path):
    """
    Performs demo operation using trained NLI model. Either takes two strings or list of strings. Compares the
    premise - hypothesis pairs and returns the NLI result.
    :param premise: opinion sentence.
    :param hypothesis: opinion sentence.
    :param transformer_type: type of the transformer in this case it is 'bert_embeddings'.
    :param nli_type: type of the nli set which the model trained on.
    :param model_type: type of the model. either ESIM or Decomposable attention.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param attention_map: boolean value to show attention heatmap of premise - hypothesis comparison.
    :param result_path: path of the file where the results will be saved.
    :return: None
    """
    print("Loading NLI model")
    model = load_model(configs[model_type] + "model")
    print("NLI model loaded")

    tokenizer = tokenization.FullTokenizer(
        vocab_file=configs[transformer_type] + "vocab.txt", do_lower_case=True)

    if type(premise) and type(hypothesis) is str:

        model.summary()
        model = Model(inputs=model.input,
                      outputs=[model.output, model.get_layer('sum_x1').output, model.get_layer('sum_x2').output])

        print("premise:", premise)
        print("hypothesis:", hypothesis)

        premise_features, premise_token = convert_examples_to_features(text=premise, max_length=max_length,
                                                                       tokenizer=tokenizer,
                                                                       attention_heatmap=True)

        hypothesis_features, hypothesis_token = convert_examples_to_features(text=hypothesis, max_length=max_length,
                                                                             tokenizer=tokenizer,
                                                                             attention_heatmap=True)

        outputs = model.predict([premise_features, hypothesis_features])
        scores = outputs[0]

        print("Entailment type is:", entailment_types[scores.argmax()],
              "\nEntailment confidence is: ", scores.max(),
              "\nContradiction score is", float("{:.4f}".format(float(outputs[0][0][1]))),
              "\nEntailment score is", float("{:.4f}".format(float(outputs[0][0][0]))),
              "\nNeutral score is,", float("{:.4f}".format(float(outputs[0][0][2]))))
        if attention_map:
            attention_visualization(tokens1=premise_token, tokens2=hypothesis_token,
                                    attention1=outputs[1], attention2=outputs[2],
                                    results_path=result_path, transformer_type=transformer_type)

    elif type(premise) and type(hypothesis) is list:

        model.summary()
        model = Model(inputs=model.input, outputs=model.output)

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
            premise_features = convert_examples_to_features(text=text1, max_length=max_length,
                                                            tokenizer=tokenizer, attention_heatmap=False)

            hypothesis_features = convert_examples_to_features(text=text2, max_length=max_length,
                                                               tokenizer=tokenizer, attention_heatmap=False)

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

        print("Total Contradiction = ", float("{:.2f}".format(float(contradiction / total))))
        print("Total Entailment =", float("{:.2f}".format(float(entailment / total))))
        print("Total Neutral =", float("{:.2f}".format(float(neutral / total))))

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
        evaluate(test_loc=args.test_loc,
                 max_length=args.max_length,
                 transformer_type=args.transformer_type,
                 model_type=args.model_type)

    elif args.mode == "demo":

        # path = "/media/ulgen/Samsung/contradiction_data_depo/results/a/data/UKPConvArg1Strict-XML/"

        # premise, _ = xml_test_file_reader(path=path + "christianity-or-atheism-_atheism.xml")
        # hypothesis, _ = xml_test_file_reader(path=path + "christianity-or-atheism-_christianity.xml")

        premise = "in the park alice plays a flute solo"
        hypothesis = "someone playing music outside"

        demo(premise=premise,
             hypothesis=hypothesis,
             transformer_type=args.transformer_type,
             nli_type=args.nli_type,
             model_type=args.model_type,
             max_length=args.max_length,
             attention_map=args.visualization,
             result_path=args.result_path
             )


if __name__ == "__main__":
    plac.call(main)
