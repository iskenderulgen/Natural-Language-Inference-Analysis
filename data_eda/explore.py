import json
import os
import pickle
import pandas as pd
from keras.utils import to_categorical
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from scipy import spatial
import numpy as np
import argparse

from bert import extract_features

path = "/home/ulgen/Documents/Python_Projects/Contradiction/data/"

# conf = SparkConf().setMaster("local[*]") \
#     .setAppName("Contradiction Pre Process") \
#     .set("spark.rdd.compress", "true") \
#     .set("spark.executor.memory", "40g")
#
# sc = SparkContext(conf=conf)
# sc.setLogLevel("ERROR")
# spark = SparkSession \
#     .builder \
#     .config(conf=conf) \
#     .getOrCreate()

# df = pd.read_json(path + 'Processed_SNLI/train/neutral.json')
# print(df)


# with open(path + 'train_x.pkl', 'rb') as f:
#     train_X = pickle.load(f)
#     for row in train_X:
#         print(row)

LABELS = {"entailment": 0, "contradiction": 1, "neutral": 2}
parser = argparse.ArgumentParser()
parser.add_argument('--bert_directory', type=str, default=path + "bert/",
                    help='main directory for the bert files')
args = parser.parse_args()
np.set_printoptions(precision=6)


def read_snli(path):
    texts1 = []
    texts2 = []
    labels = []
    with open(path, "r") as file_:
        for line in file_:
            eg = json.loads(line)
            label = eg["gold_label"]
            if label == "-":  # per Parikh, ignore - SNLI entries
                continue
            texts1.append(eg["sentence1"])
            texts2.append(eg["sentence2"])
            labels.append(LABELS[label])
    return texts1, texts2, to_categorical(np.asarray(labels, dtype="int32"))


def train():
    text, hypotesis, dev_labels = read_snli(path + 'SNLI/snli_dev.jsonl')
    sents = text + hypotesis
    print(len(text))
    print(len(hypotesis))
    print(len(sents))
    print(type(text))
    print(type(hypotesis))
    print(type(sents))

    vecs = extract_features.main(args=args, input_file=sents)
    print(vecs.shape)

    for i in range(3):
        print(vecs[i])

    print(len(vecs[:9842]))
    print(len(vecs[9842:]))


# train()

pd.set_option('display.max_colwidth', -1)

# arr = [[0.106217, 0.377535, -0.598523, -0.18559, 0.448664], [0.248715, 0.784982, -0.344282, -0.393607, -0.148429]]
# print(arr[0])


# df = pd.read_json(path+'Processed_SNLI/train/neutral.json')['sentence1_vectors'].to_numpy()
# print(type(df))
#
# with open(path + 'df.pkl', 'wb') as f:
#     pickle.dump(df, f)
# arr = [[0.106217, 0.377535, -0.598523, -0.18559, 0.448664], [0.248715, 0.784982, -0.344282, -0.393607, -0.148429]]
# arr2 = [[0.106217, 0.377535, -0.598523, -0.18559, 0.448664], [0.248715, 0.784982, -0.344282, -0.393607, -0.148429]]
#
# df1 = pd.DataFrame(data={'A': arr})
# df1.insert(loc=1, column='_similarity', value=pd.DataFrame(data={'A': arr2}), allow_duplicates=True)
# print(df1)
#
# df1.to_json(path + "aaa.json", orient='records')
#
# x = pd.read_json(path_or_buf=path + "aaa.json", orient='records')
#
# with pd.option_context('display.max_rows', None, 'display.max_columns', 30):
#     print(x)

if os.path.isfile(path=path + "Processed_SNLI/Bert_Processed/train_x.pkl"):
    print("Pre-Processed train file is found now loading")
    with open(path + 'Processed_SNLI/Bert_Processed/train_x.pkl', 'rb') as f:
        train_x = pickle.load(f)

print(np.asarray(train_x).shape)
print(np.asarray(train_x).shape[0])
print(np.asarray(train_x[0]).shape)

total = np.asarray(train_x[0])
input_a = total[:9842]
input_b = total[9842:]
print(np.asarray(input_a).shape)
print(np.asarray(input_b).shape)
print(train_x)
# b = [np.array(total[: 9842]), np.array(total[9842:])]
# print(np.asarray(b).shape)
# with open(path + 'Processed_SNLI/Bert_Processed/dev_x.pkl', 'wb') as f:
#     pickle.dump(b, f, protocol=pickle.HIGHEST_PROTOCOL)
