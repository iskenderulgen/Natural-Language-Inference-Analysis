import json
import pickle

from keras.utils import to_categorical
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from spacy.util import ensure_path

"""Pandas numpy transformation."""
# df = pd.read_json(path+'Processed_SNLI/train/neutral.json')['sentence1_vectors'].to_numpy()
# print(type(df))


"""array to pandas data frame example."""
# arr = [[0.106217, 0.377535, -0.598523, -0.18559, 0.448664], [0.248715, 0.784982, -0.344282, -0.393607, -0.148429]]
# arr2 = [[0.106217, 0.377535, -0.598523, -0.18559, 0.448664], [0.248715, 0.784982, -0.344282, -0.393607, -0.148429]]
# df1 = pd.DataFrame(data={'A': arr})
# df1.insert(loc=1, column='_similarity', value=pd.DataFrame(data={'A': arr2}), allow_duplicates=True)

"""pandas write dataframe to json"""
# df1.to_json(path + "aaa.json", orient='records')
# x = pd.read_json(path_or_buf=path + "aaa.json", orient='records')

"""Pandas show non-truncated results"""
# with pd.option_context('display.max_rows', None, 'display.max_columns', 30):

"""Spacy show the unique vector size/count."""
# print(len(nlp_fasttext.vocab.vectors))

"""while using hand made pre-trained models. get the word's index """
# word_ids.append(num_unk + vocab.get(token.text))

"""read file and index to rows / token one by one """
# def indexing_lines_astokens(fName):
#     d = {}
#     with open(fName, 'r') as f:
#         content = f.readlines()
#         lnc = 0
#         result = {}
#         for line in content:
#             print(len(content))
#             line = line.rstrip()
#             words = line.split(" ")
#             for word in words:
#                 tmp = result.get(word)
#                 if tmp is None:
#                     result[word] = []
#                 if lnc not in result[word]:
#                     result[word].append(lnc)
#
#             lnc = lnc + 1
#
#         return result

"""Load vocabulary in to dictionary"""

# def load_vocab(vocab_file):
#     """Loads a vocabulary file into a dictionary."""
#     vocab = collections.OrderedDict()
#     index = 0
#     with tf.gfile.GFile(vocab_file, "r") as reader:
#         while True:
#             token = convert_to_unicode(reader.readline())
#             if not token:
#                 break
#             token = token.strip()
#             vocab[token] = index
#             index += 1
#     return vocab

"""Turn glove raw file to token - vector pair while not using spacy"""

# vocab_txt = open(path + "Processed_SNLI/Glove_Processed/vocab.txt", "w")
# vectors = []
#
# for line in f:
#     values = line.split(sep=" ")
#     vocab_txt.write(values[0])
#     vocab_txt.write("\n")
#     vectors.append(np.asarray(values[1:], dtype='float32'))
# f.close()
# vocab_txt.close()


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
#
# path = "/media/ulgen/Samsung/contradiction_data/SNLI/snli_test.jsonl"
#
#
# def data_read_preparation(main_path):
#     dataframe = spark.read.json(main_path)
#     print("Total rows in dataframe = ", dataframe.count())
#     print(dataframe.show(5))
#
#     dataframe.createOrReplaceTempView(name="SNLI")
#     spark.sql(
#         "SELECT sentence1, gold_label, sentence2 FROM SNLI ").show(200, truncate=False)
#
# data_read_preparation(main_path=path)
# WHERE gold_label == 'entailment'
LABELS = {"entailment": 0, "contradiction": 1, "neutral": 2}

path = "/media/ulgen/Samsung/contradiction_data/SNLI/"

# def read_snli(path):
#     texts1 = []
#     texts2 = []
#     labels = []
#     loc = ensure_path(path)
#     limit = None
#     if loc.parts[-1].endswith("snli_train.jsonl"):
#         limit = 60000
#     elif loc.parts[-1].endswith("snli_dev.jsonl"):
#         limit = 1000
#     elif loc.parts[-1].endswith("snli_test.jsonl"):
#         limit = 1000
#     count = 0
#     with open(path, "r") as file_:
#         for line in file_:
#             while count < limit:
#                 eg = json.loads(line)
#                 label = eg["gold_label"]
#                 if label == "-":  # ignore - SNLI entries
#                     continue
#                 texts1.append(eg["sentence1"])
#                 texts2.append(eg["sentence2"])
#                 labels.append(LABELS[label])
#                 count = count + 1
#     return texts1, texts2, to_categorical(np.asarray(labels, dtype="int32"))
#
# t1,t2,lab = read_snli(path+"snli_dev.jsonl")
# print(len(t1))


with open("/media/ulgen/Samsung/contradiction_data/Processed_SNLI/bert_sentence/dev_x.pkl", "rb") as f:
    train_x = pickle.load(f)
print(np.asarray(train_x).shape)
print()

#print(np.asarray(train_x).shape)
# new = np.asarray(train_x).squeeze(axis=2)
# final = [np.array(new[: 549367]), np.array(new[549367:])]
#
# with open("/media/ulgen/Samsung/contradiction_data/Processed_SNLI/bert_sentence/train_x.pkl", "wb") as f:
#     pickle.dump(new, f, protocol=pickle.HIGHEST_PROTOCOL)
