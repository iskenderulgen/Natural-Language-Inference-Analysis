import argparse

import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from scipy import spatial
import matplotlib.pyplot as plt
from data_eda import extract_features

conf = SparkConf().setMaster("local[*]") \
    .setAppName("Contradiction Pre Process") \
    .set("spark.rdd.compress", "true")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
spark = SparkSession \
    .builder \
    .config(conf=conf) \
    .getOrCreate()

path = "/home/ulgen/Documents/Python_Projects/Contradiction/data/"

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='snli_dev.jsonl',
                    help='SNLI main data type - snli_train.jsonl / snli_dev.jsonl / snli_test.jsonl')
parser.add_argument('--data_type', type=str, default='dev/',
                    help='purpose of the data type, train - test or dev')
parser.add_argument('--bert_directory', type=str, default=path + "bert/",
                    help='main directory for the bert files')

args = parser.parse_args()


def data_splitter(main_path, dataset, data_type):
    dataframe = spark.read.json(main_path + dataset)
    print("Total rows in dataframe = ", dataframe.count())
    print(dataframe.show(5))

    dataframe.createOrReplaceTempView(name="SNLI")
    entailment_data = spark.sql(
        "SELECT sentence1, gold_label, sentence2 FROM SNLI WHERE gold_label == 'entailment'").toPandas()
    neutral_data = spark.sql(
        "SELECT sentence1, gold_label, sentence2 FROM SNLI WHERE gold_label == 'neutral'").toPandas()
    contradiction_data = spark.sql(
        "SELECT sentence1, gold_label, sentence2 FROM SNLI WHERE gold_label == 'contradiction'").toPandas()

    entailment_sent1 = entailment_data['sentence1'].to_numpy()
    entailment_sent1_vectors = extract_features.main(args=args, input_file=entailment_sent1)
    entailment_sent2 = entailment_data['sentence2'].to_numpy()
    entailment_sent2_vectors = extract_features.main(args=args, input_file=entailment_sent2)
    frame = similarity(entailment_sent1_vectors, entailment_sent2_vectors)
    entailment_data.insert(3, 'similarity', frame, allow_duplicates=True)
    data_save(entailment_data, main_path=path, data_type=args.data_type, label_definition="entailment")

    neutral_sent1 = neutral_data['sentence1'].to_numpy()
    neutral_sent1_vectors = extract_features.main(args=args, input_file=neutral_sent1)
    neutral_sent2 = neutral_data['sentence2'].to_numpy()
    neutral_sent2_vectors = extract_features.main(args=args, input_file=neutral_sent2)
    frame = similarity(neutral_sent1_vectors, neutral_sent2_vectors)
    neutral_data.insert(3, 'similarity', frame, allow_duplicates=True)
    data_save(neutral_data, main_path=path, data_type=args.data_type, label_definition="neutral")

    contradiction_sent1 = contradiction_data['sentence1'].to_numpy()
    contradiction_sent1_vectors = extract_features.main(args=args, input_file=contradiction_sent1)
    contradiction_sent2 = contradiction_data['sentence2'].to_numpy()
    contradiction_sent2_vectors = extract_features.main(args=args, input_file=contradiction_sent2)
    frame = similarity(contradiction_sent1_vectors, contradiction_sent2_vectors)
    contradiction_data.insert(3, 'similarity', frame, allow_duplicates=True)
    data_save(contradiction_data, main_path=path, data_type=args.data_type, label_definition="contradiction")

    contradiction = spark.read.json(path + "dev/contradiction.json").withColumnRenamed('similarity',
                                                                                       'contradiction_similarity')
    entailment = spark.read.json(path + "dev/entailment.json").withColumnRenamed('similarity',
                                                                                 'entailment_similarity')
    neutral = spark.read.json(path + "dev/neutral.json").withColumnRenamed('similarity', 'neutral_similarity')

    plotting(contradiction=contradiction, entailment=entailment, neutral=neutral)


def similarity(array1, array2):
    sim = []
    for i in range(len(array1)):
        entailment_vectors_sim = 1 - spatial.distance.cosine(array1[i], array2[i])
        sim.append(entailment_vectors_sim)
    print(sim)
    sim_column = pd.DataFrame(data=sim, index=None, columns=None)
    # print(sim_column.shape)
    # print(sim_column)
    return sim_column


def data_save(data, main_path, data_type, label_definition):
    data.to_json(main_path + data_type + label_definition + ".json", orient='records')


def plotting(contradiction, entailment, neutral):
    contradiction.show(5)
    entailment.show(5)
    neutral.show(5)

    contra_sim = contradiction.select("contradiction_similarity").toPandas().to_numpy()
    entail_sim = entailment.select("entailment_similarity").toPandas().to_numpy()
    neutral_sim = neutral.select("neutral_similarity").toPandas().to_numpy()

    fig1, ax1 = plt.subplots()
    ax1.set_title('contradiction_similarity')
    ax1.boxplot(contra_sim)

    fig2, ax1 = plt.subplots()
    ax1.set_title('entailment_similarity')
    ax1.boxplot(entail_sim)

    fig3, ax1 = plt.subplots()
    ax1.set_title('neutral_similarity')
    ax1.boxplot(neutral_sim)

    plt.show()


data_splitter(main_path=path, dataset=args.dataset, data_type=args.data_type)
