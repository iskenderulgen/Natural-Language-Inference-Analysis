import argparse

import numpy as np
import plac
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from bert import tokenization
from utilities.utils import read_nli, attention_visualization, xml_test_file_reader, load_configurations, \
    predictions_to_html

configs = load_configurations()
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="evaluate",
                    help="This argument is to select whether to carry out 'evaluate' or 'demo' operation. Evaluate"
                         "operation takes labeled test data and measures the accuracy of the model. Demo operation"
                         "is for comparing unlabeled data. Demo support two individual sentences or list of sentences"
                         "as input data.")

parser.add_argument("--nli_type", type=str, default="snli",
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

parser.add_argument("--bert_tf_hub_path", type=str, default=configs["transformer_paths"]["tf_hub_path"],
                    help="import bert tensforlow hub model")

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


def bert_encode(text, max_length, tokenizer, attention_heatmap):
    text = tokenizer.tokenize(text)

    text = text[:max_length - 2]
    input_sequence = ["[CLS]"] + text + ["[SEP]"]
    pad_len = max_length - len(input_sequence)

    tokens = tokenizer.convert_tokens_to_ids(input_sequence)
    tokens += [0] * pad_len
    pad_masks = [1] * len(input_sequence) + [0] * pad_len
    segment_ids = [0] * max_length

    if attention_heatmap:
        return np.asarray(tokens), np.asarray(pad_masks), np.asarray(segment_ids), input_sequence
    else:
        return np.asarray(tokens), np.asarray(pad_masks), np.asarray(segment_ids)


def evaluate(dev_loc, max_length, tf_hub_path, model_path, model_type):
    print("Loading trained NLI model")
    model = load_model(model_path[model_type] + "model.h5", custom_objects={"tf": tf, "KerasLayer": hub.KerasLayer})
    print("trained NLI model loaded")

    bert_encoder = hub.KerasLayer(tf_hub_path)
    vocab_file = bert_encoder.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_encoder.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    model.summary()
    model = Model(inputs=model.input,
                  outputs=[model.output, model.get_layer('sum_x1').output, model.get_layer('sum_x2').output])

    premise, hypothesis, dev_labels = read_nli(dev_loc)

    total = 0.0
    true_p = 0.0

    for text1, text2, label in zip(premise, hypothesis, dev_labels):
        tokens1, masks1, segments1 = bert_encode(text=text1, max_length=max_length,
                                                 tokenizer=tokenizer, attention_heatmap=False)

        tokens2, masks2, segments2 = bert_encode(text=text2, max_length=max_length,
                                                 tokenizer=tokenizer, attention_heatmap=False)

        outputs = model.predict([[tokens1], [masks1], [segments1],
                                 [tokens2], [masks2], [segments2]])
        # scores = outputs[0]
        if entailment_types[outputs[0].argmax()] == entailment_types[label.argmax()]:
            true_p += 1
        total += 1
    print("Entailment Model Accuracy is", true_p / total)


def demo(premise, hypothesis, transformer_type, model_path, model_type,
         max_length, attention_map, result_path, nli_type, tf_hub_path):
    """
    Performs demo operation using trained NLI model. Either takes two strings or list of strings and compares the
    premise - hypothesis pairwise and returns the NLI result.
    :param tf_hub_path: Bert contextualized transformer Tensorflow Hub Path.
    :param premise: opinion sentence
    :param hypothesis: opinion sentence
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
    model = load_model(model_path[model_type] + "model.h5", custom_objects={"tf": tf, "KerasLayer": hub.KerasLayer})
    print("NLI model loaded")

    bert_encoder = hub.KerasLayer(tf_hub_path)
    vocab_file = bert_encoder.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_encoder.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    model.summary()
    model = Model(inputs=model.input,
                  outputs=[model.output, model.get_layer('sum_x1').output, model.get_layer('sum_x2').output])

    if type(premise) and type(hypothesis) is str:

        print("premise:", premise)
        print("hypothesis:", hypothesis)

        tokens1, masks1, segments1, words1 = bert_encode(text=premise, max_length=max_length,
                                                         tokenizer=tokenizer, attention_heatmap=True)

        tokens2, masks2, segments2, words2 = bert_encode(text=hypothesis, max_length=max_length,
                                                         attention_heatmap=True, tokenizer=tokenizer)

        outputs = model.predict([[tokens1], [masks1], [segments1],
                                 [tokens2], [masks2], [segments2]])
        scores = outputs[0]

        print("Entailment type is:", entailment_types[scores.argmax()],
              "\nEntailment confidence is: ", scores.max(),
              "\nContradiction score is", float("{:.8f}".format(float(outputs[0][0][1]))),
              "\nEntailment score is", float("{:.8f}".format(float(outputs[0][0][0]))),
              "\nNeutral score is,", float("{:.8f}".format(float(outputs[0][0][2]))))

        if attention_map:
            attention_visualization(tokens1=words1, tokens2=words2,
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
            tokens1, masks1, segments1, words1 = bert_encode(text=text1, max_length=max_length,
                                                             tokenizer=tokenizer, attention_heatmap=False)

            tokens2, masks2, segments2, words2 = bert_encode(text=text2, max_length=max_length,
                                                             attention_heatmap=False, tokenizer=tokenizer)

            outputs = model.predict([[tokens1], [masks1], [segments1],
                                     [tokens2], [masks2], [segments2]])
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
        print("test mode is", args.mode)
        evaluate(dev_loc=args.test_loc,
                 max_length=args.max_length,
                 tf_hub_path=args.bert_tf_hub_path,
                 model_path=args.model_save_path,
                 model_type=args.model_type)

    elif args.mode == "demo":
        print("test mode is", args.mode)

        # path = "/media/ulgen/Samsung/contradiction_data_depo/results/a/data/UKPConvArg1Strict-XML/"

        # premise, _ = xml_test_file_reader(path=path + "christianity-or-atheism-_atheism.xml")
        # hypothesis, _ = xml_test_file_reader(path=path + "christianity-or-atheism-_christianity.xml")

        premise = "in the park alice plays a flute solo"
        hypothesis = "someone playing music outside"

        demo(max_length=args.max_length,
             tf_hub_path=args.bert_tf_hub_path,
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
