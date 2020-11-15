import argparse
import os

import matplotlib.pyplot as plt
import plac
import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras import models
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
from utilities.utils import read_nli, load_configurations

print("if tf runs on eager mode:", tf.executing_eagerly())

configs = load_configurations()
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="decomposable_attention",
                    help="Type of the model that will be trained. "
                         "for ESIM model type 'esim' "
                         "for decomposable attention model type 'decomposable_attention'. ")

parser.add_argument("--transformer_path", type=str, default=configs["transformer_paths"],
                    help="transformer model path which will convert the text in to word-ids and vectors. "
                         "Transformer path has four sub paths. transformer_type will load the desired nlp object.")

parser.add_argument("--train_loc", type=str, default=configs["nli_set_train"],
                    help="Train data location which will be processed via transformers and saved to 'processed_path'"
                         "location.")

parser.add_argument("--dev_loc", type=str, default=configs["nli_set_dev"],
                    help="Train dev data location which will be used to measure train accuracy while training model,"
                         "files will be processed using transformer and saved to 'processed_path' location.")

parser.add_argument("--max_length", type=str, default=configs["max_length"],
                    help="Max length of the sentences, longer sentences will be pruned and shorter ones will be padded"
                         "Remember longer sentences means longer sequences to train. Select best length based"
                         "on your rig.")

parser.add_argument("--model_save_path", type=str, default=configs["model_paths"],
                    help="The path where the trained NLI model will be saved.")

parser.add_argument("--batch_size", type=int, default=configs["batch_size"],
                    help="Batch size of model, it represents the amount of data the model will train for each pass.")

parser.add_argument("--nr_epoch", type=int, default=configs["nr_epoch"],
                    help="Total number of times that model will iterate trough the data.")

parser.add_argument("--nr_hidden", type=int, default=configs["nr_hidden"],
                    help="Hidden neuron size of the model")

parser.add_argument("--nr_class", type=int, default=configs["nr_class"],
                    help="Number of class that will model classify the data into. Also represents the last layer of"
                         "the model.")

parser.add_argument("--learning_rate", type=float, default=configs["learn_rate"],
                    help="Learning rate parameter that represents the constant which will be multiplied with the data"
                         "in each back propagation")

parser.add_argument("--result_path", type=str, default=configs["results"],
                    help="path where trained model loss and accuracy graphs will be saved.")

parser.add_argument("--early_stopping", type=int, default=configs["early_stopping"],
                    help="early stopping parameter for model, which stops training when reaching best accuracy.")
args = parser.parse_args()


def bert_encode(texts, tokenizer, max_len):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tf.convert_to_tensor(tokens))
        all_masks.append(tf.convert_to_tensor(pad_masks))
        all_segments.append(tf.convert_to_tensor(segment_ids))

    return all_tokens, all_masks, all_segments


def decomp_model(bert_layer, max_length, nr_hidden, nr_class, learning_rate):
    input_word_ids_1 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/0')
    input_mask_1 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/1')
    segment_ids_1 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/2')

    input_word_ids_2 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/3')
    input_mask_2 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/4')
    segment_ids_2 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/5')

    _, x1 = bert_layer([input_word_ids_1, input_mask_1, segment_ids_1])
    _, x2 = bert_layer([input_word_ids_2, input_mask_2, segment_ids_2])

    F = create_feedforward(num_units=nr_hidden)
    att_weights = layers.dot([F(x1), F(x2)], axes=-1)

    G = create_feedforward(num_units=nr_hidden)

    norm_weights_a = layers.Lambda(normalizer(1), name='normalize_axis_1_of_att_weights')(att_weights)
    norm_weights_b = layers.Lambda(normalizer(2), name='normalize_axis_2_of_att_weights')(att_weights)
    alpha = layers.dot([norm_weights_a, x1], axes=1, name='dot_product_norm_weight_a_with_x1')
    beta = layers.dot([norm_weights_b, x2], axes=1, name='dot_product_norm_weight_b_with_x2')

    # step 2: compare
    comp1 = layers.concatenate([x1, beta], name='concatenate_x1_with_beta')
    comp2 = layers.concatenate([x2, alpha], name='concatenate_x2_with_alpha')
    x1 = layers.TimeDistributed(G, name='Time_Distribute_concat_1_with_feed_forward_G')(comp1)
    x2 = layers.TimeDistributed(G, name='Time_Distribute_concat_2_with_feed_forward_G')(comp2)

    # step 3: aggregate
    v1_sum = layers.Lambda(sum_word, name='sum_x1')(x1)
    v2_sum = layers.Lambda(sum_word, name='sum_x2')(x2)
    concat = layers.concatenate([v1_sum, v2_sum])

    H = create_feedforward(num_units=nr_hidden)
    out = H(concat)
    out = layers.Dense(nr_class, activation="softmax", name='last_classifier_layer', dtype=tf.float32)(out)

    model = Model(inputs=[[input_word_ids_1, input_mask_1, segment_ids_1],
                          [input_word_ids_2, input_mask_2, segment_ids_2]],
                  outputs=[out])

    model.summary()

    model.compile(
        optimizer=optimizers.Adam(lr=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def create_feedforward(num_units, activation="relu", dropout_rate=0.2):
    return models.Sequential(
        [
            layers.Dense(num_units, activation=activation),
            layers.Dropout(dropout_rate),
            layers.Dense(num_units, activation=activation),
            layers.Dropout(dropout_rate),
        ]
    )


def normalizer(axis):
    def _normalize(att_weights):
        exp_weights = K.exp(att_weights)
        sum_weights = K.sum(exp_weights, axis=axis, keepdims=True)
        return exp_weights / sum_weights

    return _normalize


def sum_word(x):
    return K.sum(x, axis=1)


def train_model(model_save_path, model_type, max_length, batch_size, nr_epoch,
                nr_hidden, nr_class, learning_rate, early_stopping,
                result_path):
    bert_path = "/media/ulgen/Samsung/contradiction_data_depo/transformers/tf_hub"
    bert_layer = hub.KerasLayer(bert_path, trainable=False)

    train_texts1, train_texts2, train_labels = read_nli(args.train_loc)
    dev_texts1, dev_texts2, dev_labels = read_nli(args.dev_loc)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    train_all_tokens1, train_all_masks1, train_all_segments1 = bert_encode(train_texts1, tokenizer, max_len=max_length)
    train_all_tokens2, train_all_masks2, train_all_segments2 = bert_encode(train_texts2, tokenizer, max_len=max_length)

    dev_all_tokens1, dev_all_masks1, dev_all_segments1 = bert_encode(dev_texts1, tokenizer, max_len=max_length)
    dev_all_tokens2, dev_all_masks2, dev_all_segments2 = bert_encode(dev_texts2, tokenizer, max_len=max_length)

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                       patience=early_stopping, restore_best_weights=True)

    model = decomp_model(bert_layer=bert_layer, max_length=max_length, nr_hidden=nr_hidden, nr_class=nr_class,
                         learning_rate=learning_rate)

    history = model.fit([train_all_tokens1,
                         train_all_masks1,
                         train_all_segments1,
                         train_all_tokens2,
                         train_all_masks2,
                         train_all_segments2],
                        train_labels,
                        validation_data=(
                            [dev_all_tokens1,
                             dev_all_masks1,
                             dev_all_segments1,
                             dev_all_tokens2,
                             dev_all_masks2,
                             dev_all_segments2], dev_labels),
                        epochs=nr_epoch,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=[es])

    if not os.path.isdir(model_save_path[model_type]):
        os.mkdir(model_save_path[model_type])
    print("Saving trained model to", model_save_path[model_type])

    model.save(model_save_path[model_type] + "model.h5")

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

    print('\n model history:', history.history)


def main():
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    #             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
    #     except RuntimeError as e:
    #         print(e)

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    train_model(model_save_path=args.model_save_path,
                model_type=args.model_type,
                max_length=args.max_length,
                batch_size=args.batch_size,
                nr_epoch=args.nr_epoch,
                nr_hidden=args.nr_hidden,
                nr_class=args.nr_class,
                learning_rate=args.learning_rate,
                early_stopping=args.early_stopping,
                result_path=args.result_path)


if __name__ == "__main__":
    plac.call(main)
