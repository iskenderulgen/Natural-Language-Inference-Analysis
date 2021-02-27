"""
This script is for spacy based NLI model that trained with pretrained word weights such as glove - fasttext - word2vec.
Script works as standalone, loads the model and carries out the prediction.
"""
import argparse

import numpy as np
import plac
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

from utilities.utils import read_nli, load_spacy_nlp, attention_visualization, load_configurations, \
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

parser.add_argument("--transformer_type", type=str, default="glove",
                    help="Type of the transformer which will convert texts in to word-ids. Also carries the path "
                         "information of transformer object. Currently three types are supported."
                         " Here the types as follows 'glove' -  'fasttext' - 'word2vec' - 'ontonotes5'.")

parser.add_argument("--model_type", type=str, default="decomposable_attention",
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

parser.add_argument("--nr_unk", type=int, default=configs["nr_unk"],
                    help="number of unknown vectors which will be used for padding the short sentences to desired"
                         "length. Nr unknown vectors will be created using random module")

parser.add_argument("--test_loc", type=str, default=configs["nli_set_test"],
                    help="Test data location which will be used to measure the accuracy of the model")

parser.add_argument("--result_path", type=str, default=configs["results"],
                    help="path of the folder where results and graphs will be saved.")
args = parser.parse_args()

entailment_types = ["entailment", "contradiction", "neutral"]


def convert_examples_to_features(text, max_length, nlp, num_unk, attention_heatmap):
    doc = nlp(str(text))

    word_ids = []
    for i, token in enumerate(doc):
        if token.has_vector and token.vector_norm == 0:
            continue
        if i > max_length:
            break
        if token.has_vector:
            word_ids.append(token.rank + num_unk + 1)
        else:
            # if we don't have a vector, pick an OOV entry
            word_ids.append(token.rank % num_unk + 1)

    word_id_vec = np.zeros(max_length, dtype="int")
    clipped_len = min(max_length, len(word_ids))
    word_id_vec[:clipped_len] = word_ids[:clipped_len]
    word_id_vec = np.asarray(word_id_vec).reshape((1, max_length))

    if attention_heatmap:
        def get_attended_tokens(doc):
            words = []
            for token in doc:
                words.append(token.text)
            return words

        tokens = get_attended_tokens(doc=doc)

        return word_id_vec, tokens
    else:
        return word_id_vec


def evaluate(test_loc, max_length, transformer_type, model_type, num_unk):
    premise, hypothesis, dev_labels = read_nli(test_loc)
    nlp = load_spacy_nlp(configs=configs, transformer_type=transformer_type)

    #################################
    # idx = np.random.choice(np.arange(len(dev_labels)), 5000, replace=False)
    # premise = np.array(premise)[idx.astype(int)]
    # hypothesis = np.array(hypothesis)[idx.astype(int)]
    # dev_labels = np.array(dev_labels)[idx.astype(int)]
    #################################

    print("Loading trained NLI model")
    model = load_model(configs[model_type] + "model")
    print("trained NLI model loaded")
    model.summary()
    model = Model(inputs=model.input, outputs=model.output)

    total = 0.0
    true_p = 0.0

    for text1, text2, label in zip(premise, hypothesis, dev_labels):
        premise_features = convert_examples_to_features(text=text1, max_length=max_length, nlp=nlp,
                                                        num_unk=num_unk, attention_heatmap=False)

        hypothesis_features = convert_examples_to_features(text=text2, max_length=max_length, nlp=nlp,
                                                           num_unk=num_unk, attention_heatmap=False)

        outputs = model.predict([premise_features, hypothesis_features])
        # scores = outputs[0]
        if entailment_types[outputs[0].argmax()] == entailment_types[label.argmax()]:
            true_p += 1
        total += 1
    print("NLI Model Accuracy is:", true_p / total)


def demo(premise, hypothesis, transformer_type, model_type, max_length, num_unk, attention_map, result_path, nli_type):
    nlp = load_spacy_nlp(configs=configs[transformer_type], transformer_type=transformer_type)

    print("Loading NLI model")
    model = load_model(configs[model_type] + "model")
    print("NLI model loaded")

    if type(premise) and type(hypothesis) is str:
        print("premise:", premise)
        print("hypothesis:", hypothesis)

        premise_features, premise_token = convert_examples_to_features(text=premise, max_length=max_length, nlp=nlp,
                                                                       num_unk=num_unk, attention_heatmap=True)

        hypothesis_features, hypothesis_token = convert_examples_to_features(text=hypothesis, max_length=max_length,
                                                                             nlp=nlp, num_unk=num_unk,
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
            premise_features = convert_examples_to_features(text=text1, max_length=max_length, nlp=nlp,
                                                            num_unk=num_unk, attention_heatmap=False)

            hypothesis_features = convert_examples_to_features(text=text2, max_length=max_length,
                                                               nlp=nlp, num_unk=num_unk,
                                                               attention_heatmap=False)

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
                 model_type=args.model_type,
                 num_unk=args.nr_unk)

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
             num_unk=args.nr_unk,
             attention_map=args.visualization,
             result_path=args.result_path
             )


if __name__ == "__main__":
    plac.call(main)
