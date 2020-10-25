"""
This script is for spacy based NLI model that trained with pretrained word weights such as glove - fasttext - word2vec.
Script works as standalone, loads the model and carries out the prediction.
"""
import argparse
import numpy as np
import plac
import tensorflow as tf

from keras import Model
from keras.models import load_model
from utils.utils import read_nli, load_spacy_nlp, attention_visualization, load_configurations, xml_test_file_reader, \
    predictions_to_html

try:
    import cPickle as pickle
except ImportError:
    import pickle

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

parser.add_argument("--transformer_type", type=str, default="glove",
                    help="Type of the transformer which will convert texts in to word-ids. Currently three types "
                         "are supported.Here the types as follows 'glove' -  'fasttext' - 'word2vec'."
                         "Pick one you'd like to transform into")

parser.add_argument("--transformer_path", type=str, default=configs["transformer_paths"],
                    help="Main transformer model path which will convert the text in to word-ids and vectors. "
                         "transformer path has four sub paths, load_nlp module will carry out the sub model paths"
                         "based on transformer_type selection")

parser.add_argument("--max_length", type=str, default=configs["max_length"],
                    help="Max length of the sentences,longer sentences will be pruned and shorter ones will be zero"
                         "padded. Remember longer sentences means longer sequences to train. Select best length based"
                         "on your rig.")

parser.add_argument("--model_save_path", type=str, default=configs["model_paths"],
                    help="The path where trained NLI model is saved.")

parser.add_argument("--model_type", type=str, default="esim",
                    help="Type of the model that will be trained. "
                         "for ESIM model type 'esim' "
                         "for decomposable attention model type 'decomposable_attention'. ")

parser.add_argument("--result_path", type=str, default=configs["results"],
                    help="path of the file where results and graphs will be saved.")

parser.add_argument("--nr_unk", type=int, default=configs["nr_unk"],
                    help="number of unknown vectors which will be used for padding the short sentences to desired"
                         "length.Nr unknown vectors will be created using random module")

parser.add_argument("--test_loc", type=str, default=configs["nli_set_test"],
                    help="Test data location which will be used to measure the evaluation accuracy,")

parser.add_argument("--visualization", type=bool, default=False,
                    help="shows attention heatmaps between two opinion sentences, best used with single"
                         "premise- hypothesis opinion sentences.")
args = parser.parse_args()

entailment_types = ["entailment", "contradiction", "neutral"]


def get_word_ids(docs, max_length=100, nr_unk=100):
    xs = np.zeros((len(docs), max_length), dtype="int32")
    for i, doc in enumerate(docs):
        for j, token in enumerate(doc):
            if j == max_length:
                break
            if token.has_vector:
                xs[i, j] = token.rank + nr_unk + 1
            else:
                xs[i, j] = token.rank % nr_unk + 1
    return xs


class SpacyPrediction(object):
    entailment_types = ["entailment", "contradiction", "neutral"]

    @classmethod
    def load(cls, path, max_length, get_features=None):
        if get_features is None:
            get_features = get_word_ids

        model = load_model(path, custom_objects={"tf": tf})
        print("loading model")
        #############
        model = Model(inputs=model.input,
                      outputs=[model.output,
                               model.get_layer('sum_x1').output,
                               model.get_layer('sum_x2').output])
        #############
        model.summary()
        print("NLI model loaded")

        return cls(model, get_features=get_features, max_length=max_length)

    def __init__(self, model, get_features=None, max_length=100):
        self.model = model
        self.get_features = get_features
        self.max_length = max_length

    def __call__(self, doc):
        doc.user_hooks["similarity"] = self.predict
        doc.user_span_hooks["similarity"] = self.predict

        return doc

    def predict(self, doc1, doc2):
        x1 = self.get_features([doc1], max_length=self.max_length)
        x2 = self.get_features([doc2], max_length=self.max_length)
        outputs = self.model.predict([x1, x2])

        # scores = outputs[0]
        # return self.entailment_types[scores.argmax()], scores.max(), outputs[1], outputs[2]

        return outputs


def evaluate(dev_loc, max_length, transformer_path, transformer_type, model_path, model_type):
    """
    This function is to measure model accuracy, it takes labeled test NLI data and performs model test on it.
    :param dev_loc: labeled test data location.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param transformer_path: path of the transformer nlp object.
    :param transformer_type: type of the transformer glove - fasttext or word2vec.
    :param model_path: path where the model is saved as h5 file.
    :param model_type: type of the model. either ESIM or Decomposable attention.
    :return: None
    """

    disabled_pipelines = ['parser', 'tagger', 'ner', 'textcat']

    premise, hypothesis, dev_labels = read_nli(dev_loc)
    nlp = load_spacy_nlp(transformer_path=transformer_path, transformer_type=transformer_type)
    nlp.add_pipe(SpacyPrediction.load(path=model_path[model_type] + "model.h5", max_length=max_length))

    total = 0.0
    true_p = 0.0

    for text1, text2, label in zip(premise, hypothesis, dev_labels):
        doc1 = nlp(text1, disable=disabled_pipelines)
        doc2 = nlp(text2, disable=disabled_pipelines)

        outputs = doc1.similarity(doc2)
        scores = outputs[0]
        if entailment_types[scores.argmax()] == SpacyPrediction.entailment_types[label.argmax()]:
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
    :param transformer_path: path of the transformer nlp object.
    :param transformer_type: type of the transformer glove - fasttext or word2vec.
    :param model_path: path where the model is saved as h5 file.
    :param model_type: type of the model. either ESIM or Decomposable attention.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param attention_map: boolean value to show attention heatmap of string based comparison.
    :param result_path: path of the file where the results will be saved.
    :param nli_type: type of the nli set which the model trained on.
    :return: None
    """

    disabled_pipelines = ['parser', 'tagger', 'ner', 'textcat']
    nlp = load_spacy_nlp(transformer_path=transformer_path, transformer_type=transformer_type)
    nlp.add_pipe(SpacyPrediction.load(path=model_path[model_type] + "model.h5", max_length=max_length))

    if type(premise) and type(hypothesis) is str:

        doc1 = nlp(premise, disable=disabled_pipelines)
        doc2 = nlp(hypothesis, disable=disabled_pipelines)

        print("premise:", doc1)
        print("hypothesis   :", doc2)

        outputs = doc1.similarity(doc2)
        scores = outputs[0]
        print("Entailment type is:", entailment_types[scores.argmax()], "\nEntailment confidence is: ", scores.max())

        if attention_map:
            def get_attended_tokens(doc):
                words = []
                for token in doc:
                    words.append(token.text)
                return words

            tokens1 = get_attended_tokens(doc=doc1)
            tokens2 = get_attended_tokens(doc=doc2)

            attention_visualization(tokens1=tokens1,
                                    tokens2=tokens2,
                                    attention1=outputs[1],
                                    attention2=outputs[2],
                                    results_path=result_path,
                                    transformer_type=transformer_type)

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
            doc1 = nlp(text1, disable=disabled_pipelines)
            doc2 = nlp(text2, disable=disabled_pipelines)

            outputs = doc1.similarity(doc2)
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
