import os
import pickle
import numpy as np

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

from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('/media/ulgen/Samsung/contradiction_data/GoogleNews-vectors-negative300.bin', binary=True)