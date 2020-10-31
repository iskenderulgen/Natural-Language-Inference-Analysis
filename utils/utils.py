import glob
import importlib
import json
import os
import pickle

import en_core_web_lg
import matplotlib.pyplot as plt
import numpy as np
import plac
import seaborn as sns
import spacy
import yaml
import xml.etree.ElementTree as ET
import pandas as pd

from keras import backend as K
from keras.utils import to_categorical
from pathlib import Path

"""Pandas show non-truncated results"""
pd.option_context('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 2000)

LABELS = {"entailment": 0, "contradiction": 1, "neutral": 2}


def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ["KERAS_BACKEND"] = backend
        importlib.reload(K)
        assert K.backend() == backend
    if backend == "tensorflow":
        K.get_session().close()
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.per_process_memory_fraction = 0.8
        cfg.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=cfg))
        K.clear_session()


def load_configurations():
    """
    Loads the configuration yaml file that contains all the paths and neural network parameters. Majority of the
    settings are handled in configuration yaml file. This enables single centralized control over parameters and paths.
    :return: returns configurations.
    """
    with open("/home/ulgen/Documents/Python_Projects/Contradiction/configurations.yaml", 'r') as stream:
        try:
            configurations = (yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

        return configurations


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
            labels.append(LABELS[label])
    print("NLI dataset loaded")
    return texts1, texts2, to_categorical(np.asarray(labels, dtype="int32"))


def load_spacy_nlp(transformer_path, transformer_type):
    """
    Loads spacy based NLP object which converts raw text into ids of tokens. Currently this project supports three
    types of NLP embeddings objects. Glove - Fasttext - Word2Vec
    :param transformer_path: path of the transformer object.
    :param transformer_type: type definition of the transformer.
    :return: returns NLP object.
    """
    nlp = None

    if transformer_type == 'glove':
        print("Loading Glove Vectors")
        spacy.prefer_gpu()
        gpu = spacy.require_gpu()
        print("GPU:", gpu)
        nlp = en_core_web_lg.load()

    elif transformer_type == 'fasttext':
        print("Loading fasttext Vectors")
        spacy.prefer_gpu()
        gpu = spacy.require_gpu()
        print("GPU:", gpu)
        nlp = spacy.load(transformer_path[transformer_type])

    elif transformer_type == 'word2vec':
        print("Loading word2vec Vectors")
        spacy.prefer_gpu()
        gpu = spacy.require_gpu()
        print("GPU:", gpu)
        nlp = spacy.load(transformer_path[transformer_type])

    """shows the unique vector size/count."""
    # print(transformer_type, "unique vector size / count", len(nlp.vocab.vectors))
    return nlp


def attention_visualization(tokens1, tokens2, attention1, attention2, results_path, transformer_type):
    """
    Function to draw attention heatmap of the prediction scores. Takes two sentences and their attention values
    corresponding to their tokens. Then dot products the scores to achieve attention scores and draws them to a
    graph. Note that it only takes the real tokens and their attention values not the padded scores.
    :param tokens1: tokens of the sentence one.
    :param tokens2: tokens of the sentence two.
    :param attention1: attention scores of the first sentence.
    :param attention2: attention scores of the second sentence.
    :param results_path: result path where the heatmap will be saved.
    :param transformer_type: type of the NLP transformer.
    :return: None.
    """
    sentence1_length = len(tokens1)
    sentence2_length = len(tokens2)

    attentions_scores = []

    for i in attention1[0][:sentence1_length]:
        for j in attention2[0][:sentence2_length]:
            attentions_scores.append(np.dot(i, j))
    attentions_scores = np.asarray(attentions_scores) / np.sum(attentions_scores)

    plt.subplots(figsize=(10, 10))

    ax = sns.heatmap(attentions_scores.reshape((sentence1_length, sentence2_length)), linewidths=0.5, annot=False,
                     cbar=True, cmap="Blues")

    ax.set_yticklabels([i for i in tokens1])
    plt.yticks(rotation=0)
    ax.set_xticklabels([j for j in tokens2])
    plt.xticks(rotation=90)
    plt.title("attention visualized with " + transformer_type)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(results_path + 'attention_graph.png')


def predictions_to_html(nli_type, premises, hypothesises, prediction, contradiction_score, neutral_score,
                        entailment_score, result_path):
    """
    Writes prediction results to html file as table. This method provides easy to see approach for test results.
    Takes premise - hypothesis and their prediction label along with scores for each label.
    :param nli_type: Definition of the NLI train set which the model trained on.
    :param premises: opinion sentence
    :param hypothesises: opinion sentence
    :param prediction: predicted label of the given premise and hypothesis sentences.
    :param contradiction_score: contradiction score of the predicted label.
    :param neutral_score: neutral score of the predicted label.
    :param entailment_score: entailment score of the predicted label.
    :param result_path: path where the html file will be saved.
    :return: None.
    """
    premises.append("### last prediction result corresponds total amount of data. ###")
    hypothesises.append("### last row corresponds to total amount of each type of predicted label. ###")

    predictions_df = pd.DataFrame(
        data={'premise': premises,
              'hypothesis': hypothesises,
              nli_type + ' prediction': prediction,
              nli_type + ' contradiction score': contradiction_score,
              nli_type + ' neutral score': neutral_score,
              nli_type + ' entailment score': entailment_score}
    )
    html = predictions_df.to_html(float_format=lambda x: '%.3f' % x)
    text_file = open(result_path + nli_type + ".html", "w")
    text_file.write(html)
    text_file.close()


def find_differences(html_results_main_path, df1_set_definer, df2_set_definer, df3_set_definer):
    """
    This function is hard coded and doesn't provide any modular structure. This function will be reworked to provide
    comparison regardless of the dataframe size. Right now this function provides result only for three different
    result files.
    :param html_results_main_path: main path of the html results that will be compared.
    :param df1_set_definer: train set definition of the model that predicted the results.
    :param df2_set_definer: train set definition of the model that predicted the results.
    :param df3_set_definer: train set definition of the model that predicted the results.
    :return: None
    """
    df1 = pd.read_html(html_results_main_path + "/" + df1_set_definer + ".html")
    df2 = pd.read_html(html_results_main_path + "/" + df2_set_definer + ".html")
    df3 = pd.read_html(html_results_main_path + "/" + df3_set_definer + ".html")

    premises = []
    hypothesis = []

    df1_prediction = []
    df1_entailment = []
    df1_contradiction = []
    df1_neutral = []

    df2_prediction = []
    df2_entailment = []
    df2_contradiction = []
    df2_neutral = []

    df3_prediction = []
    df3_entailment = []
    df3_contradiction = []
    df3_neutral = []

    for i in range(len(df1[0])):
        if (str(df1[0][df1_set_definer + ' prediction'][i]) != str(df2[0][df2_set_definer + ' prediction'][i]) or
                str(df1[0][df1_set_definer + ' prediction'][i]) != str(df3[0][df3_set_definer + ' prediction'][i]) or
                str(df2[0][df2_set_definer + ' prediction'][i]) != str(df3[0][df3_set_definer + ' prediction'][i])):

            premises.append(str(df1[0]['premise'][i]))
            hypothesis.append(str(df1[0]['hypothesis'][i]))

            df1_prediction.append(df1[0][df1_set_definer + ' prediction'][i])
            df1_contradiction.append(df1[0][df1_set_definer + ' contradiction score'][i])
            df1_neutral.append(df1[0][df1_set_definer + ' neutral score'][i])
            df1_entailment.append(df1[0][df1_set_definer + ' entailment score'][i])

            df2_prediction.append(df2[0][df2_set_definer + ' prediction'][i])
            df2_contradiction.append(df2[0][df2_set_definer + ' contradiction score'][i])
            df2_neutral.append(df2[0][df2_set_definer + ' neutral score'][i])
            df2_entailment.append(df2[0][df2_set_definer + ' entailment score'][i])

            df3_prediction.append(df3[0][df3_set_definer + ' anli prediction'][i])
            df3_contradiction.append(df3[0][df3_set_definer + ' contradiction score'][i])
            df3_neutral.append(df3[0][df3_set_definer + ' neutral score'][i])
            df3_entailment.append(df3[0]['snli mnli anli entailment score'][i])

    merged_df = pd.DataFrame(
        data={'premise': premises,
              'hypothesis': hypothesis,

              df1_set_definer + ' prediction': df1_prediction,
              df1_set_definer + ' entailment score': df1_entailment,
              df1_set_definer + ' contradiction score': df1_contradiction,
              df1_set_definer + ' neutral score': df1_neutral,

              df2_set_definer + ' multiclass prediction': df2_prediction,
              df2_set_definer + ' entailment score': df2_entailment,
              df2_set_definer + ' contradiction score': df2_contradiction,
              df2_set_definer + ' neutral score': df2_neutral,

              df3_set_definer + ' multiclass prediction': df3_prediction,
              df3_set_definer + ' entailment score': df3_entailment,
              df3_set_definer + ' contradiction score': df3_contradiction,
              df3_set_definer + ' neutral score': df3_neutral})

    html = merged_df.to_html()
    text_file = open(html_results_main_path + "/differences.html", "w")
    text_file.write(html)
    text_file.close()


def xml_test_file_reader(path):
    """
    Reads the XML based paired data. Original data acquired from the research named:
    'What makes a convincing argument? Empirical analysis and
     detecting attributes of convincingness in Web argumentation'
     Data is paired opinion sentences around 16 types of topic. It can present contradiction and entailment paris
     based on the selection of the data.
    :param path: path of the XML file.
    :return: list of the paired sentences.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    def find_text_in_tree(arg_number):
        arg_text = []
        for item in root.findall('annotatedArgumentPair/' + arg_number):
            text = item.find('text').text
            arg_text.append(text.replace("\n", " "))

        return arg_text

    arg1_text = find_text_in_tree(arg_number='arg1')
    arg2_text = find_text_in_tree(arg_number='arg2')

    return arg1_text, arg2_text


def write_nli_to_disk(data, nli_set_path, nli_definition):
    """
    writes nli dataset passed from 'anli_to_snli' - 'mnli_to_snli' - 'merge_snli_style_sets', to disk. Uses original
    SNLI jsonl format.
    :param data: data that will be written to the drive.
    :param nli_set_path: path of the snli. this path is used for both existing and new nli set.
    :param nli_definition: definition of the nli set. 'train' - 'dev' - 'test'.
    :return: None
    """
    path = ('/'.join(Path(nli_set_path).parts[:-1]) + '/new_' + nli_definition + '.jsonl')

    with open(path, "w") as outfile:
        for line in data:
            outfile.write(json.dumps(line) + "\n")


def anli_to_snli(nli_set_path, nli_definition):
    """
    Converts ANLI dataset to SNLI format. anli uses different label structure. this code converts them to SNLI format.
    anli has 3 dataset named R1 - R2 - R3. if one wants to have total ANLI dataset, use merge function to merge all
    three set to achieve ANLI dataset.
    :param nli_set_path: path of the nli. this path is used for both existing and new nli set.
    :param nli_definition: definition of the nli set. 'train' - 'dev' - 'test'.
    :return: None
    """
    total_data = []

    with open(nli_set_path, "r") as file_:
        for line in file_:
            data = {}
            eg = json.loads(line)
            data["sentence1"] = (eg["context"])
            data["sentence2"] = (eg["hypothesis"])
            if eg["label"] == "n":
                data["gold_label"] = "neutral"
            elif eg["label"] == "c":
                data["gold_label"] = "contradiction"
            elif eg["label"] == "e":
                data["gold_label"] = "entailment"
            total_data.append(data)

    write_nli_to_disk(data=total_data, nli_set_path=nli_set_path, nli_definition=nli_definition)


def mnli_to_snli(nli_set_path):
    """
    converts MNLI to SNLI format. MNLI comes with train - dev matched - dev mismatched. Matched comes from train
    distribution while mismatched comes from unseen data. We extract first 10k examples for test. second 10k examples
    for dev and rest goes for train. We use matched - mismatched data for evaluation after train.
    :param nli_set_path: path of the MNLI data.
    :return: None
    """
    total_data = []

    with open(nli_set_path, "r") as file_:
        for line in file_:
            data = {}
            eg = json.loads(line)
            label = eg["gold_label"]
            if label == "-":  # ignore - MNLI entries
                continue
            data["sentence1"] = (eg["sentence1"])
            data["sentence2"] = (eg["sentence2"])
            data["gold_label"] = eg["gold_label"]
            total_data.append(data)

    write_nli_to_disk(data=total_data[0:10000], nli_set_path=nli_set_path, nli_definition="test")
    write_nli_to_disk(data=total_data[10000:20000], nli_set_path=nli_set_path, nli_definition="dev")
    write_nli_to_disk(data=total_data[20000:], nli_set_path=nli_set_path, nli_definition="train")


def merge_snli_style_sets(nli_set_path, nli_definition):
    """
    Merges all nli set that found in the path. Beware, all the sets must be converted to snli format before merging.
    Function reads all data sequentially and merges them, saves as snli format. This approach usually used to see
    the model behaviour when trained on all nli sets.
    :param nli_set_path: folder path of the nli sets.
    :param nli_definition: train - test - dev definition of the nli data. saves the merged set based on this definer.
    :return: None
    """
    total_data = []

    for filename in glob.glob(nli_set_path):
        with open(os.path.join(os.cwd(), filename), 'r') as file_:
            for line in file_:
                data = {}
                eg = json.loads(line)
                label = eg["gold_label"]
                if label == "-":  # ignore - MNLI entries
                    continue
                data["sentence1"] = (eg["sentence1"])
                data["sentence2"] = (eg["sentence2"])
                data["gold_label"] = eg["gold_label"]
                total_data.append(data)

    write_nli_to_disk(data=total_data, nli_set_path=nli_set_path, nli_definition=nli_definition)


def convert_glove_tokens_weights(glove_path):
    """
    Takes original glove pretrained txt file and splits it to tokens - weights file. Vocab.txt will be used to
    create word-id matrix for embedding layer. After having token-id vector, one can feed those files to keras
    embedding layer to train network. When creating token-id matrix, to get the id from dictionary
    use 'vocab.get(token.text)'. With newest enhancements, we recommend using spacy module to create NLP objects
    and create token-id matrix.
    :param glove_path: glove file path.
    :return: None
    """
    vectors = []
    vocab_txt = open('/'.join(Path(glove_path).parts[:-1]) + '/vocab.txt', "w")
    with open(glove_path, "r") as glove:
        for line in glove:
            values = line.split(sep=" ")
            vocab_txt.write(values[0])
            vocab_txt.write("\n")
            vectors.append(np.asarray(values[1:], dtype='float32'))

    glove.close()
    vocab_txt.close()

    with open('/'.join(Path(glove_path).parts[:-1]) + "/weights.pkl", 'wb') as f:
        pickle.dump(vectors, f)


def main():
    """
    NLI files manipulations can be handled using utils main.
    :return: None
    """
    pass


if __name__ == "__main__":
    plac.call(main)
