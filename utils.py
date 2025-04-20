import json
import pandas as pd
import os


def load_nli_data(file_path):
    """
    Load and filter NLI data from a JSONL file

    Args:
        file_path: Path to the NLI JSONL file

    Returns:
        A filtered pandas DataFrame with sentence1, sentence2, and label columns
    """
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    # Filter out examples with '-' gold label
    df = df[df["gold_label"] != "-"]

    # Map labels to integers
    df["label"] = df["gold_label"].map(
        {"entailment": 0, "contradiction": 1, "neutral": 2}
    )

    # Keep only relevant columns
    df = df[["sentence1", "sentence2", "label", "gold_label"]]

    return df




# TODO: Clean the following codes
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