"""
This code coverts premises and hypothesis using pre trained initial bert word weights. Process is straight forward and
similar to pre-trained approaches. Instead of prediction based approach we extract initial-word-matrix from bert model
and use this to create id's of sentences. Thanks to bert's full-tokenizer we can achieve good results with
35,522 token & vector.
"""

import argparse
import datetime
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import plac
import tensorflow as tf
from bert import tokenization
from tensorflow.keras.callbacks import EarlyStopping

from models.decomposable_attention import decomposable_attention_model
from models.esim import esim_bilstm_model
from utilities.utils import read_nli, load_configurations, set_memory_growth

set_memory_growth()
configs = load_configurations()
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="esim",
                    help="Type of the model architecture that the model is trained on. This parameter also carries the "
                         "model save path information hence this is used for both defining architecture and carrying "
                         "path information."
                         "for ESIM model use 'esim' "
                         "for Decomposable Attention model use 'decomposable_attention'.")

parser.add_argument("--transformer_type", type=str, default="bert_embeddings",
                    help="Type of the transformer which will convert texts in to word-ids. Also carries the path "
                         "information of transformer object. This script is designed for only bert actual embeddings."
                         "Parameter takes only 'bert_embeddings' option.")

parser.add_argument("--train_loc", type=str, default=configs["nli_set_train"],
                    help="Train data location which will be processed with BERT Tf-Hub and fed in to downstream task.")

parser.add_argument("--dev_loc", type=str, default=configs["nli_set_dev"],
                    help="Dev data location which will be used to measure train accuracy while training model,"
                         "files will be processed with BERT Tf-Hub and fed in to downstream task.")

parser.add_argument("--test_loc", type=str, default=configs["nli_set_test"],
                    help="Test data location which will be used to measure model accuracy after the training model,"
                         "files will be processed with NLP object and fed in to downstream task.")

parser.add_argument("--max_length", type=str, default=configs["max_length"],
                    help="Max length of the sentences, longer sentences will be pruned and shorter ones will be zero"
                         "padded. longer sentences mean longer sequences to train. Pick best length based on your rig.")

parser.add_argument("--processed_path", type=str, default=configs["processed_nli"],
                    help="Path where the transformed texts will be saved as word-ids. Word-id matrix will be used"
                         "in embedding layer to retrieve word weights from lookup table.")

parser.add_argument("--batch_size", type=int, default=configs["batch_size"],
                    help="Batch size of model, it represents the amount of data the model will train for each pass.")

parser.add_argument("--nr_epoch", type=int, default=configs["nr_epoch"],
                    help="Total number of times that model will iterate trough whole data.")

parser.add_argument("--nr_hidden", type=int, default=configs["nr_hidden"],
                    help="Hidden neuron size of the model")

parser.add_argument("--nr_class", type=int, default=configs["nr_class"],
                    help="Number of classes that will model classify the data into. Also represents the last layer of"
                         "the model.")

parser.add_argument("--learning_rate", type=float, default=configs["learn_rate"],
                    help="Learning rate parameter which will update the weights on back propagation")

parser.add_argument("--early_stopping", type=int, default=configs["early_stopping"],
                    help="early stopping parameter for model, which stops training when reaching best accuracy.")

parser.add_argument("--result_path", type=str, default=configs["results"],
                    help="path of the folder where trained model loss and accuracy graphs will be saved.")
args = parser.parse_args()


def convert_examples_to_features(premises, hypothesis, max_length, transformer_type):
    """
    This function uses bert transformer to create word-ids of tokens. This process is similar to pre-trained word
    weight transformation.This method uses bert's actual word weights to create vectors. Compared to priors, 
    bert actual word weights are much smaller than old pre-trained vectors.
    :param premises: opinion sentence
    :param hypothesis: opinion sentence
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param transformer_type: path of the BERT embeddings transformer object.
    :return: returns word ids as a list.
    """

    start_time = datetime.datetime.now()
    sentences = premises + hypothesis
    print("Total sentences to be processed:", len(sentences))
    processed_sent_count = 0
    features = []

    tokenizer = tokenization.FullTokenizer(
        vocab_file=transformer_type + "vocab.txt", do_lower_case=True)

    for example in sentences:
        tokens = tokenizer.tokenize(example)

        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        input_ids_raw = tokenizer.convert_tokens_to_ids(tokens)

        while len(input_ids_raw) < max_length:
            input_ids_raw.append(0)
        assert len(input_ids_raw) == max_length

        features.append(input_ids_raw)

        processed_sent_count = processed_sent_count + 1
        if processed_sent_count % 5000 == 0:
            print("processed Sentence: %d Processed Percentage: %.2f" %
                  (processed_sent_count, processed_sent_count / len(sentences) * 100))

    print("Total time spent to create token ID's of sentences: ", datetime.datetime.now() - start_time)

    return [np.array(features[: len(premises)]), np.array(features[len(premises):])]


def extract_initial_word_embedding_matrix(file_name, tensor_name, all_tensors=False, all_tensor_names=False):
    """
    This function exports BERT's actual word weights from the tensor model. This weights will be used to create
    token-ids - weights matrix to be used in embedding layer. Currently bert contains 35.552 token and weights.
    :param file_name: bert pre-trained model file name
    :param tensor_name: name of the tensor which will be extracted
    :param all_tensors: whether to print all tensors or not.
    :param all_tensor_names: whether to print all tensor names or not.
    :return: returns word weights that extracted from bert model.
    """

    embeds = []

    reader = tf.train.load_checkpoint(file_name)
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

    bert_vector_size = np.asarray(embeds).shape[1]
    embedding_matrix = np.zeros((bert_vector_size, 1024), dtype="float32")
    embedding_matrix[0:bert_vector_size] = np.asarray(embeds)

    return embedding_matrix


def bert_pretrained_transformer(transformer_type, train_loc, dev_loc, test_loc, max_length, processed_path):
    """
    This function reads NLI sets and processes them. Takes sentences as list and transforms them in to word-id matrix.
     This word_id matrix will be then saved to disk as pkl file to be read and used in embedding layer of the madel.
    :param transformer_type: type of the transformer object and transformer identifier.
    :param train_loc: NLI train dataset location.
    :param dev_loc: NLI dev dataset location.
    :param max_length: max length of the sentence. Longer ones will be pruned shorter ones will be padded.
    :param processed_path: path where the processed files will be saved.
    :return: train - dev set as word-ids matrix and corresponding labels.
    """

    print("starting to pre-process using bert-initial word embeddings.")
    train_texts1, train_texts2, train_labels = read_nli(train_loc)
    dev_texts1, dev_texts2, dev_labels = read_nli(dev_loc)
    test_texts1, test_texts2, test_labels = read_nli(test_loc)

    if not os.path.isdir(processed_path):
        print("Processed_SNLI directory is not exist, now created")
        os.mkdir(processed_path)

    if os.path.isfile(path=processed_path + "train_x.pkl"):
        print("Pre-Processed train file is found now loading")
        with open(processed_path + "train_x.pkl", "rb") as f:
            train_x = pickle.load(f)
    else:
        print("There is no pre-processed file of train_X, pre-process will start now")
        train_x = convert_examples_to_features(premises=train_texts1, hypothesis=train_texts2, max_length=max_length,
                                               transformer_type=configs[transformer_type])
        with open(processed_path + "train_x.pkl", "wb") as f:
            pickle.dump(train_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(path=processed_path + "dev_x.pkl"):
        print("Pre-processed dev file is found now loading")
        with open(processed_path + "dev_x.pkl", "rb") as f:
            dev_x = pickle.load(f)
    else:
        print("There is no pre-processed file of dev_X, pre-process will start now")
        dev_x = convert_examples_to_features(premises=dev_texts1, hypothesis=dev_texts2, max_length=max_length,
                                             transformer_type=configs[transformer_type])
        with open(processed_path + "dev_x.pkl", "wb") as f:
            pickle.dump(dev_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(path=processed_path + "test_x.pkl"):
        print(transformer_type, "based pre processed test file is found, now loading...")
        with open(processed_path + "test_x.pkl", "rb") as f:
            test_x = pickle.load(f)
    else:
        print(transformer_type, "based pre processed file of test data isn't exist, pre process will start now.")
        test_x = convert_examples_to_features(premises=test_texts1, hypothesis=test_texts2, max_length=max_length,
                                              transformer_type=configs[transformer_type])
        with open(processed_path + "test_x.pkl", "wb") as f:
            pickle.dump(test_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(configs[transformer_type] + "weights.pkl"):
        print("Embedding matrix is already exist")
        with open(configs[transformer_type] + "weights.pkl", "rb") as f:
            word_weights = pickle.load(f)
    else:
        checkpoint_path = configs[transformer_type] + "bert_model.ckpt"
        word_weights = extract_initial_word_embedding_matrix(file_name=checkpoint_path,
                                                             tensor_name='bert/embeddings/word_embeddings',
                                                             all_tensors=False, all_tensor_names=False)
        with open(configs[transformer_type] + "weights.pkl", 'wb') as f:
            pickle.dump(word_weights, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Bert initial word weights based feature extraction and embedding matrix extraction completed.")

    return train_x, train_labels, dev_x, dev_labels, test_x, test_labels, word_weights


def train_model(model_type, max_length, batch_size, nr_epoch, nr_hidden, nr_class, learning_rate,
                early_stopping, train_x, train_labels, dev_x, test_x, test_labels, dev_labels, vectors, result_path):
    """
    Model will be trained in this function. Currently it supports ESIM and Decomposable Attention models.
    :param model_type: type of the model and model save path. Either ESIM or Decomposable Attention.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param batch_size: represents the amount of data the model will train for each pass.
    :param nr_epoch: total number of times the model iterates trough all the training data.
    :param nr_hidden: hidden neuron size of the model.
    :param nr_class: number of classes that model will classify given pairs. Also the last layer of the model.
    :param learning_rate: learning rate parameter which will update the weights on back propagation.
    :param early_stopping: parameter that stops the training based on given condition.
    :param train_x: training data.
    :param train_labels: training labels.
    :param dev_x: developer data.
    :param dev_labels: developer labels.
    :param test_x: test data.
    :param test_labels: test data labels.
    :param vectors: embedding vectors of the tokens.
    :param result_path: path where accuracy and loss graphs will be saved along with the model history.
    :return: None
    """

    model = None

    if model_type == "esim":

        model = esim_bilstm_model(vectors=vectors, max_length=max_length, nr_hidden=nr_hidden,
                                  nr_class=nr_class, learning_rate=learning_rate)

    elif model_type == "decomposable_attention":

        model = decomposable_attention_model(vectors=vectors, max_length=max_length,
                                             nr_hidden=nr_hidden, nr_class=nr_class,
                                             learning_rate=learning_rate)

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                       patience=early_stopping, restore_best_weights=True)

    history = model.fit(
        train_x,
        train_labels,
        validation_data=(dev_x, dev_labels),
        epochs=nr_epoch,
        batch_size=batch_size,
        verbose=1,
        callbacks=[es]
    )

    if not os.path.isdir(configs[model_type]):
        os.mkdir(configs[model_type])
    print("Saving to", configs[model_type])
    model.save(configs[model_type] + "model", save_format="h5")

    print('\n model history:', history.history)

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    with open(result_path + 'result_history.txt', 'w') as file:
        file.write(str(history.history))

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(result_path + 'accuracy.png', bbox_inches='tight')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(result_path + 'loss.png', bbox_inches='tight')
    plt.show()

    print("Evaluate on test data")
    results = model.evaluate(test_x, test_labels, batch_size=32)
    print("test loss, test acc:", results)

def main():
    train_x, train_labels, dev_x, dev_labels, test_x, test_labels, vectors = \
        bert_pretrained_transformer(transformer_type=args.transformer_type,
                                    train_loc=args.train_loc,
                                    dev_loc=args.dev_loc,
                                    test_loc=args.test_loc,
                                    max_length=args.max_length,
                                    processed_path=args.processed_path)

    train_model(model_type=args.model_type,
                max_length=args.max_length,
                batch_size=args.batch_size,
                nr_epoch=args.nr_epoch,
                nr_hidden=args.nr_hidden,
                nr_class=args.nr_class,
                learning_rate=args.learning_rate,
                early_stopping=args.early_stopping,
                train_x=train_x,
                train_labels=train_labels,
                dev_x=dev_x,
                dev_labels=dev_labels,
                test_x=test_x,
                test_labels=test_labels,
                vectors=vectors,
                result_path=args.result_path)


if __name__ == "__main__":
    plac.call(main)
