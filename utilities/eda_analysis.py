import argparse
import json
import os
import pickle
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plac
# import scikit_posthocs as sp
# import scipy.stats as stats
# import statsmodels.api as sm
# from bioinfokit.analys import stat
# from statsmodels.formula.api import ols
from scipy import spatial
from utilities.utils import load_configurations, read_nli

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 2000)

configs = load_configurations()
parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str,  default=configs['processed_nli'],
                    help="NLI premise and hypothesis sentences which are converted in to weight by using BERT"
                         "transformer. These are the vector representations of the opinion sentences.")

parser.add_argument("--nli_set", type=str,  default=configs['nli_set_dev'],
                    help="This file is the original NLI train or dev/test file. We use this file to extract the labels"
                         "and to calculate to example lengths.")

parser.add_argument("--weights_definition", type=str, default="snli train",
                    help="This parameters defines the weight file NLI type and purpose. Such as snli - mnli or other"
                         "types of NLI sets and train/dev/test purpose of the NLI set. This will be used to show NLI"
                         "information as plot title.")

parser.add_argument("--result_path", type=str, default=configs["results"],
                    help="path where trained graphs will be saved.")
args = parser.parse_args()


def nli_sets_example_length_analysis(nli_path, nli_definition, result_path):
    text1, text2, _ = read_nli(path=nli_path)

    premise_token_len = []
    hypothesis_token_len = []

    for premise, hypothesis in zip(text1, text2):
        premise_token_len.append(len(premise.split(" ")))
        hypothesis_token_len.append(len(hypothesis.split(" ")))

    print("total premise examples:", len(premise_token_len))
    print("total hypothesis example:", len(hypothesis_token_len))
    print("premise mean sentence length:", statistics.mean(premise_token_len))
    print("hypothesis mean sentence length:", statistics.mean(hypothesis_token_len))
    print("longest premise sentence:", max(premise_token_len))
    print("longest hypothesis sentence:", max(hypothesis_token_len))

    plt.subplots(figsize=(10, 10))
    fig1, ax1 = plt.subplots()
    ax1.set_title(nli_definition + ' example length distribution')
    plt.boxplot(x=[premise_token_len, hypothesis_token_len], labels=['premise', 'hypothesis'],
                manage_ticks=True, autorange=True, meanline=True)
    plt.savefig(result_path + 'similarity.png', bbox_inches='tight')
    plt.draw()
    plt.show()


def exploratory_data_analysis(weights_path, nli_path, result_path, weights_definition):
    """
    This function is suitable to demonstrate exploratory data analysis on sentence weights of NLI tuples. It assumes
    that NLI premises and hypothesis are vectorized using sentence based train method and saved to disk if not use
    the sentence based method to transform sentences in to weights. Function takes the weight files and divides them
    in to three groups based on their corresponding labels. Then conducts a cosine similarity between premise and
    hypothesis and plots the results.
    :param weights_path: sentence weights of NLI sentences.
    :param nli_path: label file of the weights. Labels are extracted from the original NLI.jsonl file.
    :param result_path: path where trained graphs will be saved.
    :param weights_definition: type of the NLI set. SNLI or MNLI or others. This will be used for plot title.
    :return: None
    """
    os.path.isfile(weights_path)
    print("Pre-processed weight file is found, now loading")
    with open(weights_path, "rb") as file:
        weights = pickle.load(file)

    os.path.isfile(nli_path)
    print("NLI file is found, now loading")
    nli_set = open(nli_path, "r")

    premise_weights = weights[0]
    hypothesis_weights = weights[1]

    entailment_sim = []
    contradiction_sim = []
    neutral_sim = []

    for weights1, weights2, label in zip(premise_weights, hypothesis_weights, nli_set):
        nli_data = json.loads(label)
        label = nli_data["gold_label"]
        if label == "-":  # ignore '-'  SNLI entries
            continue
        elif label == 'entailment':
            entailment_sim.append(round(float(1 - spatial.distance.cosine(weights1, weights2)), 4))
        elif label == 'contradiction':
            contradiction_sim.append(round(float(1 - spatial.distance.cosine(weights1, weights2)), 4))
        elif label == 'neutral':
            neutral_sim.append(round(float(1 - spatial.distance.cosine(weights1, weights2)), 4))

    plt.subplots(figsize=(10, 10))
    fig1, ax1 = plt.subplots()
    ax1.set_title(weights_definition + ' similarity EDA plots')
    plt.boxplot(x=[contradiction_sim, entailment_sim, neutral_sim], labels=['contradiction', 'entailment', 'neutral'],
                manage_ticks=True, autorange=True, meanline=True)
    plt.savefig(result_path + 'similarity.png', bbox_inches='tight')
    plt.draw()
    plt.show()


# def anova_analysis(result_html_path, nli_type_1, nli_type_2, nli_type_3):
#     """
#     This  function shows the model differences using the prediction results on unseen test data. function expects that
#     results html files are merged in to one file. Anova analysis takes the same label scores that acquired from three
#     different models predicted on same test data. Then conducts analysis and gives corresponding f and p scores.
#     As an example, having three models which trained with same model architecture eg. ESIM, using three different train
#     sets such as snli, snli-mnli, snli-mnli-anli. Each model conducts prediction operation on same test data. Then each
#     result html file must be merged in to one using required function from utils.
#     :param result_html_path: path of the result html file.
#     :param nli_type_1: type of the nli file which used on training the model.
#     :param nli_type_2: type of the nli file which used on training the model.
#     :param nli_type_3: type of the nli file which used on training the model.
#     :return: None
#     """
#     df1 = pd.read_html(result_html_path)
#
#     snli_model = []
#     snli_mnli_model = []
#     snli_mnli_anli_model = []
#
#     for i in range(len(df1[0])):
#         snli_model.append(df1[0][nli_type_1 + ' neutral score'][i])
#         snli_mnli_model.append(df1[0][nli_type_2 + ' neutral score'][i])
#         snli_mnli_anli_model.append(df1[0][nli_type_3 + ' neutral score'][i])
#
#     df1 = pd.DataFrame(data={'snli_model': snli_model,
#                              'snli_mnli_model': snli_mnli_model,
#                              'snli_mnli_anli_model': snli_mnli_anli_model})
#
#     print(df1.var())
#
#     fvalue, pvalue = stats.f_oneway(np.asarray(df1['snli_model']),
#                                     np.asarray(df1['snli_mnli_model']),
#                                     np.asarray(df1['snli_mnli_anli_model']))
#
#     print("Results of ANOVA test:\n The F-statistic is:", {fvalue}, "\n The p-value is:", {pvalue})
#
#     # reshape the d dataframe suitable for statsmodels package
#     d_melt = pd.melt(df1.reset_index(), id_vars=['index'], value_vars=['snli_model',
#                                                                        'snli_mnli_model',
#                                                                        'snli_mnli_anli_model'])
#
#     # replace column names
#     d_melt.columns = ['index', 'models', 'value']
#     # Ordinary Least Squares (OLS) model
#     model = ols('value ~ C(models)', data=d_melt).fit()
#     anova_table = sm.stats.anova_lm(model, typ=1)
#     print("\n#### ANOVA TABLE")
#     print(anova_table)
#
#     res = stat()
#     res.tukey_hsd(df=d_melt, res_var='value', xfac_var='models')
#     print("\n### TUKEY SHD TABLE")
#     print(res.tukey_summary)
#
#
# def sheffe_table(result_html_path, nli_type_1, nli_type_2):
#     """
#     This function suitable to use after anova analysis. sheffe analysis compares the sets which gave p score lower than
#     the threshold. Sheffe analysis shows that which of the sets that requires p score are more significant.
#     :param result_html_path: path of the result html file.
#     :param nli_type_1: type of the nli file which used on training the model.
#     :param nli_type_2: type of the nli file which used on training the model.
#     :return:
#     """
#     df1 = pd.read_html(result_html_path)
#
#     model_1 = []
#     model_2 = []
#
#     for i in range(len(df1[0])):
#         model_1.append(df1[0][nli_type_1 + ' neutral score'][i])
#         model_2.append(df1[0][nli_type_2 + ' entailment score'][i])
#
#     df2 = pd.DataFrame(data={
#         'snli_mnli_model': model_1,
#         'snli_mnli_anli_model': model_2})
#
#     print("\n### SCHEFFE TABLE")
#     df2 = df2.melt(var_name='groups', value_name='values')
#     print(sp.posthoc_scheffe(a=df2, val_col='values', group_col='groups'))


def main():
    # exploratory_data_analysis(weights_path=args.weights_path,
    #                           labels_path=args.labels_path,
    #                           result_path=args.result_path,
    #                           weights_definition=args.weights_definition)

    nli_sets_example_length_analysis(nli_path=args.nli_set,
                                     nli_definition="train",
                                     result_path=args.result_path)

    exploratory_data_analysis(weights_path=args.weights_path + "/dev_x.pkl",
                              nli_path=args.nli_set,
                              result_path=args.result_path,
                              weights_definition=args.weights_definition)


if __name__ == "__main__":
    plac.call(main)
