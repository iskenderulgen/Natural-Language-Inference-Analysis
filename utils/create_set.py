import codecs
import json
import os

import pandas
import tensorflow as tf
import collections
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import xml.etree.ElementTree as ET
import jsonlines

# conf = SparkConf().setMaster("local[*]") \
#     .setAppName("Contradiction Pre Process") \
#     .set("spark.rdd.compress", "true") \
#     .set("spark.driver.memory", "10G")
# sc = SparkContext(conf=conf)
# sc.setLogLevel("ERROR")
# spark = SparkSession \
#     .builder \
#     .config(conf=conf) \
#     .getOrCreate()


# df = pandas.read_csv("/home/ulgen/Downloads/sample_data.csv", index_col=False)
# sentence1 = list(df['sentence1'])
# sentence2 = list(df['sentence2'])
# gold_label = list(df['gold_label'])
#
# data = {}
# with codecs.getwriter("utf-8")(tf.gfile.Open("/home/ulgen/Downloads/sample_data.jsonl", "w")) as writer:
#     for text1, text2, label in zip(sentence1, sentence2, gold_label):
#         data["sentence1"] = text1
#         data["sentence2"] = text2
#         data["gold_label"] = label
#         writer.write(json.dumps(data) + "\n")

path_main = "/media/ulgen/Samsung/contradiction_data/"

total = []


# def read_snli(path):
#     with open(path, "r") as file_:
#         # with open(path_main + "total_test.jsonl", "w") as outfile:
#         for line in file_:
#             data = {}
#             eg = json.loads(line)
#             data["sentence1"] = (eg["context"])
#             data["sentence2"] = (eg["hypothesis"])
#             #data["gold_label"] = (eg["label"])
#             if eg["label"] == "n":
#                 data["gold_label"] = "neutral"
#             elif eg["label"] == "c":
#                 data["gold_label"] = "contradiction"
#             elif eg["label"] == "e":
#                 data["gold_label"] = "entailment"
#             # outfile.write(json.dumps(data) + "\n")
#             total.append(data)
#
# read_snli(path=path_main + "ANLI/R3/dev.jsonl")
# #read_snli(path=path_main + "SNLI_MNLI/total_train.jsonl")
#
# with open(path_main + "ANLI/R3/total_train.jsonl", "w") as outfile:
#     for line in total:
#         outfile.write(json.dumps(line) + "\n")

# path = "/media/ulgen/Samsung/contradiction_data/SNLI_MNLI/total_test.jsonl"
#
# with open(path, "r") as file_:
#          for line in file_:
#              eg = json.loads(line)
#              print(eg)


os.mkdir("/home/ulgen/Downloads/aa/")