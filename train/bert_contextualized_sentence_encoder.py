import argparse
import os

import matplotlib.pyplot as plt
import plac
import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.layers import Input

from utilities.utils import read_nli, load_configurations, set_memory_growth

set_memory_growth()
configs = load_configurations()
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="esim",
                    help="Type of the model architecture that model is trained on. Contextualized word embeddings"
                         "are only used with ESIM model. This parameter takes only ESIM model architecture")

parser.add_argument("--bert_tf_hub_path", type=str, default=configs["transformer_paths"]["tf_hub_path"],
                    help="imports bert tensorflow-hub model")

parser.add_argument("--train_loc", type=str, default=configs["nli_set_train"],
                    help="Train data location which will be processed with BERT Tf-Hub and fed in to downstream task.")

parser.add_argument("--dev_loc", type=str, default=configs["nli_set_dev"],
                    help="Train dev data location which will be used to measure train accuracy while training model,"
                         "files will be processed with BERT Tf-Hub and fed in to downstream task.")

parser.add_argument("--max_length", type=str, default=configs["max_length"],
                    help="Max length of the sentences,longer sentences will be pruned and shorter ones will be zero"
                         "padded. longer sentences means longer sequences to train. Select best length based"
                         "on your rig.")

parser.add_argument("--model_save_path", type=str, default=configs["model_paths"],
                    help="The path where the trained NLI model will be saved.")

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
                    help="Learning rate parameter which will update the weights during training on back propagation")

parser.add_argument("--result_path", type=str, default=configs["results"],
                    help="path of the folder where trained model loss and accuracy graphs will be saved.")

parser.add_argument("--early_stopping", type=int, default=configs["early_stopping"],
                    help="early stopping parameter for model, which stops training when reaching best accuracy.")
args = parser.parse_args()


def bert_encode(texts, tokenizer, max_length):
    """
    Bert requires special pre processing before feeding information to BERT model. Each text must be converted in to
    token_id, pad_mask and segment ids. Bert takes three inputs as described and converts text in to contextualized
    word embeddings.
    :param texts: opinion sentence
    :param tokenizer: Bert tokenizer object
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :return: token_ids, masks, segments_ids.
    """

    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_length - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_length - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_length

        all_tokens.append(tf.convert_to_tensor(tokens))
        all_masks.append(tf.convert_to_tensor(pad_masks))
        all_segments.append(tf.convert_to_tensor(segment_ids))

    return all_tokens, all_masks, all_segments


def esim_model(bert_layer, max_length, nr_hidden, nr_class, learning_rate):
    """
    ESIM - Enhanced Sequential Inference Model. ESIM architecture trains a language inference model to classify premise
    and hypothesis pairs. ESIM uses chain LSTM approach to create robust NLI model. LSTMs are initialized as
    Bidirectional, thus enables model to preserver information in left-to-right and right-to-left directions.
    :param bert_layer: BERT tensorflow-hub keras-layer module.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param nr_hidden: hidden neuron size of the model.
    :param nr_class: number of classed that model will classify into. Also the last layer of the model.
    :param learning_rate: learning rate parameter which will update the weights during training on back propagation.
    :return: NLI model architecture.
    """

    bilstm1 = Bidirectional(LSTM(nr_hidden, return_sequences=True))
    bilstm2 = Bidirectional(LSTM(nr_hidden, return_sequences=True))

    input_word_ids_1 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/0')
    input_mask_1 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/1')
    segment_ids_1 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/2')

    input_word_ids_2 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/3')
    input_mask_2 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/4')
    segment_ids_2 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/5')

    pooled_output_1, _ = bert_layer([input_word_ids_1, input_mask_1, segment_ids_1])
    pooled_output_2, _ = bert_layer([input_word_ids_2, input_mask_2, segment_ids_2])

    """ Bert'in cıktısı [1, 1024] olabilir. Eğer direk sekilde alıyorsa buna gerek kalmadan direk feed edilebilir
        Daha önce 1024 geliyords direk pkl denç bunu test et gerek yoksa reshape layer'i silinebilir.
    """

    pooled_output_1 = layers.Reshape((1, 1024), input_shape=(1024,))(pooled_output_1)
    pooled_output_2 = layers.Reshape((1, 1024), input_shape=(1024,))(pooled_output_2)

    x1 = bilstm1(pooled_output_1)
    x2 = bilstm1(pooled_output_2)

    e = layers.Dot(axes=2, name="dot_product_of_x1_x2")([x1, x2])
    attend_e1 = layers.Softmax(axis=2, name="softmax_of_dot_on_axis_2")(e)
    attend_e2 = layers.Softmax(axis=1, name="softmax_of_dot_on_axis_1")(e)
    e1 = layers.Lambda(K.expand_dims, arguments={'axis': 3}, name="expand_dim_on_attend_e1")(attend_e1)
    e2 = layers.Lambda(K.expand_dims, arguments={'axis': 3}, name="expand_dim_on_attend_e2")(attend_e2)

    _x1 = layers.Lambda(K.expand_dims, arguments={'axis': 1}, name="expand_dim_on_x2")(x2)
    _x1 = layers.Multiply(name="multiply_e1_with_expanded_x2")([e1, _x1])
    _x1 = layers.Lambda(K.sum, arguments={'axis': 2}, name="sum_x1")(_x1)
    _x2 = layers.Lambda(K.expand_dims, arguments={'axis': 2}, name="expand_dim_on_x1")(x1)
    _x2 = layers.Multiply(name="multiply_e2_with_expanded_x1")([e2, _x2])
    _x2 = layers.Lambda(K.sum, arguments={'axis': 1}, name="sum_x2")(_x2)

    m1 = layers.Concatenate(name="concatenate_x1_with_attended_x1")(
        [x1, _x1, layers.Subtract()([x1, _x1]), layers.Multiply()([x1, _x1])])
    m2 = layers.Concatenate(name="concatenate_x2_with_attended_x2")(
        [x2, _x2, layers.Subtract()([x2, _x2]), layers.Multiply()([x2, _x2])])

    y1 = bilstm2(m1)
    y2 = bilstm2(m2)

    mx1 = layers.Lambda(K.max, arguments={'axis': 1}, name="k.max_of_y1")(y1)
    av1 = layers.Lambda(K.mean, arguments={'axis': 1}, name="k.mean_of_y1")(y1)
    mx2 = layers.Lambda(K.max, arguments={'axis': 1}, name="k.max_of_y2")(y2)
    av2 = layers.Lambda(K.mean, arguments={'axis': 1}, name="k.mean_of_y2")(y2)

    y = layers.Concatenate(name="concatenate_max_and_mean_of_y1_and_y2")([av1, mx1, av2, mx2])
    y = layers.Dense(1024, activation='relu')(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(512, activation='relu')(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(nr_class, activation='softmax', name='last_classifier_layer')(y)

    model = Model(inputs=[[input_word_ids_1, input_mask_1, segment_ids_1],
                          [input_word_ids_2, input_mask_2, segment_ids_2]],
                  outputs=[y])

    model.compile(
        optimizer=optimizers.Adam(lr=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# def sentence_embedding_layer():
#     return models.Sequential(
#         [
#             layers.Reshape(
#                 (1, 1024), input_shape=(1024,)
#             ),
#         ]
#     )


def train_model(model_save_path, model_type, max_length, batch_size, nr_epoch, nr_hidden, nr_class, learning_rate,
                early_stopping, result_path):
    """
     Model will be trained in this function. Contextualized sentence embeddings are only used with ESIM.
   :param model_save_path: path where the model will be saved as h5 file.
    :param model_type: type of the model. Contextualized BERT only used with ESIM
    :param max_length: max length of the sentence / sequence.
    :param batch_size: size of the train data which will be feed forwarded on each iteration.
    :param nr_epoch: total number of times the model iterates trough all the training data.
    :param nr_hidden: hidden neuron size of the model
    :param nr_class: number of classes that model will classify given pairs. Also the last layer of the model.
    :param learning_rate: learning rate parameter which will update the weights during training on back propagation.
    :param early_stopping: parameter that stops the training when the validation accuracy cant go higher.
    :param result_path: path where accuracy and loss graphs will be saved along with the model history.
    :return: None
    """

    bert_layer = hub.KerasLayer(handle=args.bert_tf_hub_path, trainable=False)

    train_texts1, train_texts2, train_labels = read_nli(args.train_loc)
    dev_texts1, dev_texts2, dev_labels = read_nli(args.dev_loc)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    train_tokens1, train_masks1, train_segments1 = bert_encode(train_texts1, tokenizer, max_length=max_length)
    train_tokens2, train_masks2, train_segments2 = bert_encode(train_texts2, tokenizer, max_length=max_length)

    dev_tokens1, dev_masks1, dev_segments1 = bert_encode(dev_texts1, tokenizer, max_length=max_length)
    dev_tokens2, dev_masks2, dev_segments2 = bert_encode(dev_texts2, tokenizer, max_length=max_length)

    model = esim_model(bert_layer=bert_layer, max_length=max_length, nr_hidden=nr_hidden, nr_class=nr_class,
                       learning_rate=learning_rate)

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                       patience=early_stopping, restore_best_weights=True)

    model.summary()

    history = model.fit([train_tokens1,
                         train_masks1,
                         train_segments1,
                         train_tokens2,
                         train_masks2,
                         train_segments2],
                        train_labels,
                        validation_data=(
                            [dev_tokens1,
                             dev_masks1,
                             dev_segments1,
                             dev_tokens2,
                             dev_masks2,
                             dev_segments2], dev_labels),
                        epochs=nr_epoch,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=[es])

    if not os.path.isdir(model_save_path[model_type]):
        os.mkdir(model_save_path[model_type])
    print("Saving trained model to", model_save_path[model_type])

    model.save(model_save_path[model_type] + "model.h5")

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


def main():
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
