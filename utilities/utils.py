import glob
import json
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import en_core_web_lg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import tensorflow as tf
import yaml
from gensim.models import KeyedVectors
from tensorflow.keras.utils import to_categorical

pd.option_context('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 2000)

LABELS = {"entailment": 0, "contradiction": 1, "neutral": 2}


def set_memory_growth():
    """
    Below config sets the optimized graphic memory usage. With TF2, memory can be allocated based on the computational
    load and releases the rest.
    :return: None
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(len(physical_devices), "Physical GPUs,", len(physical_devices), "Logical GPUs")
    except RuntimeError as e:
        print(e, "Invalid device or cannot modify virtual devices once initialized.")
        pass


def load_configurations():
    """
    Loads the YAML file that contains paths and neural network parameters.
    This enables single centralized control over parameters and paths.
    :return: configurations from YAML file.
    """
    with open("../configurations.yaml", 'r') as file_:
        try:
            configurations = (yaml.safe_load(file_))
        except yaml.YAMLError as exc:
            print(exc)

        return configurations


def read_nli(path):
    """
    Parses the SNLI dataset into sentences and target labels from JSONL file format
    :param path: SNLI dataset path.
    :return: parsed SNLI dataset.
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

    print("NLI dataset read and parsed.")
    return texts1, texts2, to_categorical(np.asarray(labels, dtype="int32"))


def read_test_data(path):
    """
    Reads the non-labeled test data. Any test dataset file format must be reformatted into SNLI JsonL format beforehand
    :param path: path to the test dataset.
    :return: parsed test dataset.
    """
    texts1 = []
    texts2 = []
    with open(path, "r") as file_:
        for line in file_:
            nli_data = json.loads(line)
            texts1.append(nli_data["premise"])
            texts2.append(nli_data["hypothesis"])

    print("NLI test dataset read and parsed.")
    return texts1, texts2


def load_spacy_nlp(configs, transformer_type):
    """
    Loads spacy NLP object that converts words to id of token. Currently, this project supports three types of NLP
    embeddings objects. Glove - Fasttext - Word2Vec
    :param configs: path of the transformer object
    :param transformer_type: type definition of the transformer.
    :return: Spacy NLP object.
    """

    pipelines = ['parser', 'tagger', 'ner', 'textcat', 'lemmatizer', 'attribute_ruler', 'tok2vec']

    if transformer_type == 'ontonotes5':
        print("Loading", transformer_type, "NLP object")
        nlp = en_core_web_lg.load(disable=pipelines)
    else:
        print(transformer_type, "NLP object is loaded")
        nlp = spacy.load(configs[transformer_type], disable=pipelines)

    print(transformer_type, "unique vector size / count", len(nlp.vocab.vectors))

    return nlp


def attention_visualization(premise, hypothesis, premise_weights, hypothesis_weights, results_path, transformer_type):
    """
    Draws attention heatmap of the tokens associated with corresponding weights that are extracted from the last layer
    of network. Thus, visualizes that how the network predicts the final label
    :param premise: words of the premise sentence
    :param hypothesis: words of the hypothesis sentence
    :param premise_weights: word weights of the premise sentence
    :param hypothesis_weights: word weights of the hypothesis
    :param results_path: path where the plotted attention map will be saved
    :param transformer_type: indicates NLP model used to calculate weights
    :return: None
    """

    premise_length = len(premise)
    hypothesis_length = len(hypothesis)
    attentions_scores = []

    for i in premise_weights[0][:premise_length]:
        for j in hypothesis_weights[0][:hypothesis_length]:
            attentions_scores.append(np.dot(i, j))
    attentions_scores = np.asarray(attentions_scores) / np.sum(attentions_scores)

    plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(attentions_scores.reshape((premise_length, hypothesis_length)),
                     linewidths=0.5,
                     annot=True,
                     cbar=True,
                     cmap="Blues")

    ax.set_yticklabels([i for i in premise])
    plt.yticks(rotation=0)
    ax.set_xticklabels([j for j in hypothesis])
    plt.xticks(rotation=90)
    plt.title("attention heatmap visualized with " + transformer_type)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(results_path + transformer_type + '_attention_graph.png')


def predictions_to_excel(nli_type, premises, hypothesises, prediction, contradiction_score, neutral_score,
                         entailment_score, result_path):
    """
    Writes prediction dataset to Excel file alongside with its predictions scores across each label. Each text pair
    is presented with prediction scores across each label. Thus, provides easy access for analyst to inspect results
    :param nli_type: indicates NLI dataset type such as SNLI - MNLI - ANLI or merged NLI set
    :param premises: premise texts
    :param hypothesises: hypothesis texts
    :param prediction: prediction label as in text
    :param contradiction_score: contradiction score of the text pair
    :param neutral_score: neutral score of the text pair
    :param entailment_score: entailment score of the text pair
    :param result_path: path where the Excel file will be saved
    :return: None
    """

    pd.DataFrame(
        data={'premise': premises,
              'hypothesis': hypothesises,
              nli_type + ' model prediction': prediction,
              nli_type + ' model contradiction score': contradiction_score,
              nli_type + ' model neutral score': neutral_score,
              nli_type + ' model entailment score': entailment_score}
    ).to_excel(result_path + nli_type + "_prediction_results.xlsx", index=False)


# TO - DO build iterate Excel file merger. keep premise - hypothesis as index and merge all prediction information
# from different excel prediction files and provide cumulitive result.


def xml_data_to_json(path1, path2):
    """
    Creates NLI formatted data from the research "DOI:10.18653/v1/D16-1129" that has opinionated sentences around 16
    topics as text pairs
    :param path1: path of the data that contains topic1's sentences.
    :param path2: path of the data that contains topic2's sentences
    :return: None
    """

    def xml_data_extractor(path, arg_number):
        """
        Reads the XML file format and extracts text pair information
        :param path: path to the XML file
        :param arg_number: argument number
        :return: parsed text data.
        """
        tree = ET.parse(path)
        root = tree.getroot()
        arg_text = []
        for item in root.findall('annotatedArgumentPair/' + arg_number):
            text = item.find('text').text
            arg_text.append(str(text).replace("\n", " "))

        return arg_text

    path = ('/'.join(Path(path1).parts[:-1]) + '/new_' + nli_type + '.jsonl')

    topic1_arg1_text = set(xml_data_extractor(path=path1, arg_number='arg1'))
    topic1_arg2_text = set(xml_data_extractor(path=path1, arg_number='arg2'))

    topic2_arg1_text = set(xml_data_extractor(path=path2, arg_number='arg1'))
    topic2_arg2_text = set(xml_data_extractor(path=path2, arg_number='arg2'))

    a = list(itertools.product(topic1_arg1_text, topic1_arg2_text))

    premise = [line[0] for line in a]
    hypothesis = [line[1] for line in a]

    entailment_df = pd.DataFrame(data={'premise': premise,
                                       'hypothesis': hypothesis,
                                       'label': 'entailment'})

    entailment_df = pd.DataFrame(data={'premise': [topic1_arg1_text, topic2_arg1_text],
                                       'hypothesis': [topic1_arg2_text, topic2_arg2_text],
                                       'label': 'entailment'})


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

    with open(path, "a") as outfile:
        for line in data:
            outfile.write(json.dumps(line) + "\n")


def anli_to_snli(nli_set_path, nli_definition):
    """
    Converts ANLI dataset to SNLI format. anli uses different label structure. this code converts them to SNLI format.
    anli has 3 dataset named R1 - R2 - R3. To achieve ANLI dataset, use merge function to merge all three sets.
    :param nli_set_path: path of the nli. this path is used for both existing and new nli set.
    :param nli_definition: definition of the nli set. 'train' - 'dev' - 'test'.
    :return: None
    """
    total_data = []

    with open(nli_set_path, "r") as file_:
        for line in file_:
            data = {}
            eg = json.loads(line)
            data["sentence1"] = str(eg["context"])
            data["sentence2"] = str(eg["hypothesis"])
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
    distribution while mismatched comes from unseen data. matched and mismatched has the snli format so we only work
    on the main dataset of mnli. We extract first 10k examples for test. second 10k examples
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
            data["sentence1"] = str(eg["sentence1"])
            data["sentence2"] = str(eg["sentence2"])
            data["gold_label"] = str(eg["gold_label"])
            total_data.append(data)

    write_nli_to_disk(data=total_data[0:10000], nli_set_path=nli_set_path, nli_definition="test")
    write_nli_to_disk(data=total_data[10000:20000], nli_set_path=nli_set_path, nli_definition="dev")
    write_nli_to_disk(data=total_data[20000:], nli_set_path=nli_set_path, nli_definition="train")


def merge_snli_style_sets(nli_set_path, nli_definition):
    """
    Merges all nli set that found in the path. Beware, all the sets must be converted to snli format before merging.
    Function reads all data sequentially and merges, saves as snli format. This approach usually used to see
    the model behaviour when trained with all nli sets. When sets are merged it will be written to parent folder due to
    the behaviour of 'write_nli_to_disk' function.
    :param nli_set_path: folder path of the nli sets.
    :param nli_definition: train - test - dev definition of the nli data. saves the merged set based on this definer.
    :return: None
    """
    total_data = []
    for file in (glob.glob(nli_set_path + "/" + "*.jsonl")):
        with open(file, 'r') as file_:
            for line in file_:
                data = {}
                eg = json.loads(line)
                label = eg["gold_label"]
                if label == "-":  # ignore - SNLI entries
                    continue
                data["sentence1"] = str(eg["sentence1"])
                data["sentence2"] = str(eg["sentence2"])
                data["gold_label"] = str(eg["gold_label"])
                total_data.append(data)

    write_nli_to_disk(data=total_data, nli_set_path=nli_set_path, nli_definition=nli_definition)


def convert_glove_tokens_weights(glove_path):
    """
    Takes original glove pretrained txt file and splits it to tokens - weights file. Vocab.txt will be used to
    create word-id matrix for embedding layer. After having token-id vector, one can feed those files to keras
    embedding layer to train network. When creating token-id matrix, to get the id from dictionary
    use 'vocab.get(token.text)'. With newest enhancements, we recommend using spacy module to create NLP objects
    and extract token-id matrix.
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


def label_comparison(human_label_path, model_result_path, nli_type):
    """
    This function compares the human labels and predicted model labels on unseen test data. This type of comparisons
    are usually required for when publishing a research paper. As an example, when a researcher runs the model on a real
    life data to see how model behaves. Meantime a group of native language researchers labels the test data by hand.
    When prediction time is finished. Researcher compares the model and human labels to see how good the model behaves.
    This function assumes the human label is text data which only contains the example number and label and model result
    path as html file which is a standard prediction demonstration format.
    :param human_label_path: text data that contains human labels
    :param model_result_path: html file that contains model prediction results
    :param nli_type: nli type which the predictor model trained on.
    :return: None
    """
    f = open(human_label_path, "r")
    text_labels = []
    for line in f:
        text_labels.append(line.split("-")[1].strip())

    df1 = pd.read_html(model_result_path)
    html_array = []
    for i in range(57):
        html_array.append(df1[0][nli_type + ' prediction'][i].strip())

    tp = 0
    total = 0

    for i in range(len(html_array)):
        print(html_array[i])
        if text_labels[i] == html_array[i]:
            tp += 1
        total += 1

    print("accuracy :", tp / total)


def convert_bin_word2vec_to_tex(path):
    """
    Before converting to txt from bin file, extract the gz file. Due to possible memory problems, reading from gz is
    not provided in this script.
    :param path: path of the bin file of word2vec
    :return: None
    """
    loc = Path(path)
    model = KeyedVectors.load_word2vec_format(loc, binary=True)
    loc = loc.with_suffix('.txt')
    model.wv.save_word2vec_format(loc)
    print("word2vec vectors are saved as txt file to:", loc)


def main():
    """
    NLI file manipulation and other needs can be carried out using utils function set. Define the function in main
    and run.
    :return: None
    """
    # merge_snli_style_sets(nli_set_path="/home/ulgen/Downloads/anli_raw/train", nli_definition="train")

    # convert_bin_word2vec_to_tex(
    #     path="/media/ulgen/Samsung/contradiction_data_depo/zips/GoogleNews-vectors-negative300.bin")

    xml_data_to_json(path1="/media/ulgen/Samsung/contradiction_data/results/evolution-vs-creation_evolution.xml",
                     path2="/media/ulgen/Samsung/contradiction_data/results/evolution-vs-creation_creation.xml",
                     nli_type="entailment")


if __name__ == "__main__":
    main()

"""

Pandas numpy transformation.
df = pd.read_json(path+'Processed_SNLI/train/neutral.json')['sentence1_vectors'].to_numpy()
print(type(df))


array to pandas data frame example
arr = [[0.106217, 0.377535, -0.598523, -0.18559, 0.448664], [0.248715, 0.784982, -0.344282, -0.393607, -0.148429]]
arr2 = [[0.106217, 0.377535, -0.598523, -0.18559, 0.448664], [0.248715, 0.784982, -0.344282, -0.393607, -0.148429]]
df1 = pd.DataFrame(data={'A': arr})
df1.insert(loc=1, column='_similarity', value=pd.DataFrame(data={'A': arr2}), allow_duplicates=True)

pandas write dataframe to json
df1.to_json(path + "aaa.json", orient='records')
x = pd.read_json(path_or_buf=path + "aaa.json", orient='records')
"""
