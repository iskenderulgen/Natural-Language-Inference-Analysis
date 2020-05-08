import argparse
from prediction_based import bert_encoder
import matplotlib.pyplot as plt
import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from scipy import spatial

conf = SparkConf().setMaster("local[*]") \
    .setAppName("Contradiction Pre Process") \
    .set("spark.rdd.compress", "true") \
    .set("spark.driver.memory", "10G")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
spark = SparkSession \
    .builder \
    .config(conf=conf) \
    .getOrCreate()

path = "/home/ulgen/Documents/Python_Projects/Contradiction/data/"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='SNLI/snli_test.jsonl',
                    help='SNLI main dataset - SNLI/snli_train.jsonl / SNLI/snli_dev.jsonl / SNLI/snli_test.jsonl')
parser.add_argument('--data_type', type=str, default='Processed_SNLI/train/',
                    help='data type, Processed_SNLI/train/ - Processed_SNLI/test/ or Processed_SNLI/dev/')
parser.add_argument('--bert_directory', type=str, default=path + "bert/",
                    help='main directory for the bert_dependencies files')
args = parser.parse_args()


def data_read_preparation(main_path, dataset):
    dataframe = spark.read.json(main_path + dataset)
    print("Total rows in dataframe = ", dataframe.count())
    print(dataframe.show(5))

    dataframe.createOrReplaceTempView(name="SNLI")
    entailment_sentences_df = spark.sql(
        "SELECT sentence1, gold_label, sentence2 FROM SNLI WHERE gold_label == 'entailment'").toPandas()
    neutral_sentences_df = spark.sql(
        "SELECT sentence1, gold_label, sentence2 FROM SNLI WHERE gold_label == 'neutral'").toPandas()
    contradiction_sentences_df = spark.sql(
        "SELECT sentence1, gold_label, sentence2 FROM SNLI WHERE gold_label == 'contradiction'").toPandas()

    get_vectors_and_sims(sentence_pairs=entailment_sentences_df, label_def='entailment')
    get_vectors_and_sims(sentence_pairs=neutral_sentences_df, label_def='neutral')
    get_vectors_and_sims(sentence_pairs=contradiction_sentences_df, label_def='contradiction')


def get_vectors_and_sims(sentence_pairs, label_def):
    premises = sentence_pairs['sentence1'].to_numpy()
    hypothesis = sentence_pairs['sentence2'].to_numpy()

    sentence_vectors = bert_encoder.sentence_transformer(bert_directory=args.bert_directory, premises=premises,
                                                         hypothesis=hypothesis, feature_type="bert_sentence")

    premises_vectors = sentence_vectors[0]
    hypothesis_vectors = sentence_vectors[1]

    sentence_pairs.insert(loc=1, column='premises_vectors', value=pd.DataFrame(data={'A': premises_vectors}),
                          allow_duplicates=True)

    sentence_pairs.insert(loc=4, column='hypothesis_vectors', value=pd.DataFrame(data={'A': hypothesis_vectors}),
                          allow_duplicates=True)

    similarity_frame = similarity(premises_vectors, hypothesis_vectors)
    sentence_pairs.insert(loc=5, column=label_def + '_similarity', value=similarity_frame, allow_duplicates=True)
    data_save(sentence_pairs, main_path=path, data_type=args.data_type, label_definition=label_def)
    print(label_def, " vectors are extracted")


def similarity(array1, array2):
    sim = []
    for i in range(len(array1)):
        entailment_vectors_sim = 1 - spatial.distance.cosine(array1[i], array2[i])
        sim.append(round(float(entailment_vectors_sim), 4))
    # print(sim)
    sim_column = pd.DataFrame(data=sim, index=None, columns=None)
    # print(sim_column.shape)
    # print(sim_column)
    return sim_column


def data_save(data, main_path, data_type, label_definition):
    data.to_json(main_path + data_type + label_definition + ".json", orient='records')


def plotting(data_dir, data_type):
    contra_sim = pd.read_json(data_dir + data_type + "contradiction.json")['contradiction_similarity'].to_numpy()
    entail_sim = pd.read_json(data_dir + data_type + "entailment.json")['entailment_similarity'].to_numpy()
    neutral_sim = pd.read_json(data_dir + data_type + "neutral.json")['neutral_similarity'].to_numpy()
    print("read")

    fig1, ax1 = plt.subplots()
    ax1.set_title(data_type + 'similarity_plots')
    plt.boxplot(x=[contra_sim, entail_sim, neutral_sim], labels=['contradiction', 'entailment', 'neutral'],
                manage_ticks=True, autorange=True, meanline=True)
    plt.savefig(path + data_type + 'Similarity.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # data_splitter(main_path=path, dataset=args.dataset)
    plotting(data_dir=path, data_type=args.data_type)
