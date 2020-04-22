import json
import os
import pickle
import tensorflow as tf
import numpy as np
from numpy import loadtxt
from tensorflow.python.tools.inspect_checkpoint import _count_total_params

main_path = '/home/ulgen/Documents/Python_Projects/Contradiction/data/Processed_SNLI/Bert_Processed_WordLevel/'
path = '/home/ulgen/Documents/Python_Projects/Contradiction/data/bert/vocab.txt'
bert_path = "/media/ulgen/Samsung/contradiction_data/data/bert/"


def lineIndex(fName):
    d = {}
    with open(fName, 'r') as f:
        content = f.readlines()
        lnc = 0
        result = {}
        for line in content:
            print(len(content))
            line = line.rstrip()
            words = line.split(" ")
            for word in words:
                tmp = result.get(word)
                if tmp is None:
                    result[word] = []
                if lnc not in result[word]:
                    result[word].append(lnc)

            lnc = lnc + 1

        return result


# print(lineIndex(path))

def len_text(fName):
    with open(fName, 'r') as f:
        content = f.readlines()
        count = 0
        for i in content:
            count = count + 1
        print(len(content))
        print(count)


def print_tensors_in_checkpoint_file2(file_name, tensor_name, all_tensors,
                                      all_tensor_names=False,
                                      count_exclude_pattern=""):
    embeds = []
    nr_unk = 100
    """Prints tensors in a checkpoint file.

    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.

    If `tensor_name` is provided, prints the content of the tensor.

    Args:
      file_name: Name of the checkpoint file.
      tensor_name: Name of the tensor in the checkpoint file to print.
      all_tensors: Boolean indicating whether to print all tensors.
      all_tensor_names: Boolean indicating whether to print all tensor names.
      count_exclude_pattern: Regex string, pattern to exclude tensors when count.
    """
    oov = np.random.normal(size=(100, 768))
    oov = oov / oov.sum(axis=1, keepdims=True)

    reader = tf.pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors or all_tensor_names:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            print("tensor_name: ", key)
            if all_tensors:
                print(reader.get_tensor(key))
    elif not tensor_name:
        print(reader.debug_string().decode("utf-8"))
    else:
        print("tensor_name: ", tensor_name)
        print(reader.get_tensor(tensor_name))
        embeds.append(reader.get_tensor(tensor_name))

    vectors = np.zeros((30522 + nr_unk, 768), dtype="float32")
    vectors[0: nr_unk, ] = oov
    vectors[nr_unk:30522 + nr_unk, ] = np.asarray(embeds)
    print(vectors.shape)
    print(vectors)


def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors,
                                     all_tensor_names=False,
                                     count_exclude_pattern=""):
    embeds = []
    """Prints tensors in a checkpoint file.

    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.

    If `tensor_name` is provided, prints the content of the tensor.

    Args:
      file_name: Name of the checkpoint file.
      tensor_name: Name of the tensor in the checkpoint file to print.
      all_tensors: Boolean indicating whether to print all tensors.
      all_tensor_names: Boolean indicating whether to print all tensor names.
      count_exclude_pattern: Regex string, pattern to exclude tensors when count.
    """

    reader = tf.pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors or all_tensor_names:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            print("tensor_name: ", key)
            if all_tensors:
                print(reader.get_tensor(key))
    elif not tensor_name:
        print(reader.debug_string().decode("utf-8"))
    else:
        print("tensor_name: ", tensor_name)
        # print(reader.get_tensor(tensor_name))
        embeds.append(reader.get_tensor(tensor_name))

    print(np.asarray(embeds).shape)
    print(len(embeds))
    print(embeds)


# bert/embeddings/word_embeddings

checkpoint_path = bert_path + "bert_model.ckpt"
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='cls/seq_relationship/output_weights',
                                 all_tensors=False, all_tensor_names=False)
