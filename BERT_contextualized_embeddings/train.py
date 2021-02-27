import argparse
import datetime
import os

import matplotlib.pyplot as plt
import plac
import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.layers import Input
from utilities.utils import read_nli, load_configurations, set_memory_growth
from tensorflow.keras.regularizers import l2

L2 = l2(0.0001)
tf.config.optimizer.set_jit(True)
set_memory_growth()
configs = load_configurations()
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="esim",
                    help="Type of the model architecture that the model will be trained on. Contextualized word "
                         "embeddings are only used with ESIM model. This parameter takes only ESIM model architecture. "
                         "This parameter also carries the model save path information hence this is used for both "
                         "defining architecture and carrying path information.")

parser.add_argument("--bert_tf_hub", type=str, default="bert_tf_hub_contextualized",
                    help="When used with imported config parameter from yaml file it gives the path of the "
                         "BERT tensorflow-hub model. It is also the type of the transformer,"
                         "This will also be used to label resulting graphs and other files.")

parser.add_argument("--train_loc", type=str, default=configs["nli_set_train"],
                    help="Train data location which will be processed with BERT Tf-Hub and fed in to downstream task.")

parser.add_argument("--dev_loc", type=str, default=configs["nli_set_dev"],
                    help="Dev data location which will be used to measure train accuracy while training model,"
                         "files will be processed with BERT Tf-Hub and fed in to downstream task.")

parser.add_argument("--max_length", type=str, default=configs["max_length"],
                    help="Max length of the sentences, longer sentences will be pruned and shorter ones will be zero"
                         "padded. longer sentences mean longer sequences to train. Pick best length based on your rig.")

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


def bert_encode(texts, max_length, tokenizer):
    """
    Bert requires special pre-processing before feeding information to BERT model. Each text must be converted in to
    token_id, pad_mask and segment_ids. Bert takes three inputs as described and converts text in to contextualized
    word embeddings.
    :param texts: opinion sentence.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param tokenizer: Bert tokenizer object.
    :return: token_ids, masks_ids, segments_ids.
    """
    start_time = datetime.datetime.now()
    processed_sent_count = 0
    print("Total sentences to be processed:", len(texts))

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

        processed_sent_count += 1
        if processed_sent_count % 5000 == 0:
            print("processed Sentence: %d Processed Percentage: %.2f" %
                  (processed_sent_count, processed_sent_count / len(texts) * 100))

    print("Total time spent to create token ID's of sentences:", datetime.datetime.now() - start_time)

    return all_tokens, all_masks, all_segments


def esim_model(bert_layer, max_length, nr_hidden, nr_class, learning_rate):
    """
    ESIM - Enhanced Sequential Inference Model. ESIM architecture trains a language inference model to classify premise
    and hypothesis pairs. ESIM uses chain LSTM approach to create robust NLI model. LSTMs are initialized as
    Bidirectional, thus enables model to preserver information in left-to-right and right-to-left directions.
    :param bert_layer: BERT tensorflow-hub keras-layer module.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param nr_hidden: hidden neuron size of the mid-layers.
    :param nr_class: number of label classes. Also the last layer size of the model.
    :param learning_rate: learning rate parameter which will update the weights on back propagation.
    :return: NLI model architecture.
    """

    bilstm1 = Bidirectional(LSTM(nr_hidden, return_sequences=True, recurrent_regularizer=L2))
    bilstm2 = Bidirectional(LSTM(nr_hidden, return_sequences=True, recurrent_regularizer=L2))

    input_word_ids_1 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/0')
    input_mask_1 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/1')
    segment_ids_1 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/2')

    input_word_ids_2 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/3')
    input_mask_2 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/4')
    segment_ids_2 = Input(shape=(max_length,), dtype=tf.int32, name='inputs/5')

    _, sequence_output_1 = bert_layer([input_word_ids_1, input_mask_1, segment_ids_1])
    _, sequence_output_2 = bert_layer([input_word_ids_2, input_mask_2, segment_ids_2])

    x1 = bilstm1(sequence_output_1)
    x2 = bilstm1(sequence_output_2)

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
    y = layers.Dense(1024, activation='relu', kernel_regularizer=L2)(y)
    y = layers.Dropout(0.5)(y)
    y = layers.Dense(512, activation='relu', kernel_regularizer=L2)(y)
    y = layers.Dropout(0.5)(y)
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


def decay_schedule(epoch, lr):
    if epoch == 0:
        print("epoch=", epoch, "and learning rate=", lr)
        return lr
    elif epoch != 0:
        lr = lr / 2
        print("epoch=", epoch, "and learning rate=", lr)
        return lr


def train_model(model_type, bert_tf_hub, train_data_path, dev_data_path, max_length, batch_size, nr_epoch,
                nr_hidden, nr_class, learning_rate, early_stopping, result_path):
    """
    Model will be trained in this function. Contextualized word embeddings are only used with ESIM.
    :param model_type: type of the model and model save path. Contextualized BERT only used with ESIM.
    :param bert_tf_hub: BERT contextualized transformer Tensorflow Hub module path and definition information.
    :param train_data_path: train data location.
    :param dev_data_path: dev data location.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param batch_size: represents the amount of data the model will train for each pass.
    :param nr_epoch: total number of times the model iterates trough all the training data.
    :param nr_hidden: hidden neuron size of the model.
    :param nr_class: number of classes that model will classify given pairs. Also the last layer of the model.
    :param learning_rate: learning rate parameter which will update the weights on back propagation.
    :param early_stopping: parameter that stops the training based on given condition.
    :param result_path: path where accuracy and loss graphs will be saved along with the model history.
    :return: None
    """

    bert_layer = hub.KerasLayer(handle=configs[bert_tf_hub], trainable=False)

    train_texts1, train_texts2, train_labels = read_nli(path=train_data_path)
    dev_texts1, dev_texts2, dev_labels = read_nli(path=dev_data_path)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    train_tokens1, train_masks1, train_segments1 = bert_encode(texts=train_texts1, max_length=max_length,
                                                               tokenizer=tokenizer)
    train_tokens2, train_masks2, train_segments2 = bert_encode(texts=train_texts2, max_length=max_length,
                                                               tokenizer=tokenizer)

    dev_tokens1, dev_masks1, dev_segments1 = bert_encode(texts=dev_texts1, max_length=max_length, tokenizer=tokenizer)
    dev_tokens2, dev_masks2, dev_segments2 = bert_encode(texts=dev_texts2, max_length=max_length, tokenizer=tokenizer)

    model = esim_model(bert_layer=bert_layer, max_length=max_length, nr_hidden=nr_hidden, nr_class=nr_class,
                       learning_rate=learning_rate)

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                       patience=early_stopping, restore_best_weights=True)

    mc = ModelCheckpoint(filepath=configs[model_type] + 'model.{epoch:02d}-{val_accuracy:.3f}.h5',
                         monitor='val_accuracy', save_best_only=True, mode='max')

    lr_scheduler = LearningRateScheduler(decay_schedule)

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
                        callbacks=[es, mc, lr_scheduler])

    if not os.path.isdir(configs[model_type]):
        os.mkdir(configs[model_type])
    print("Saving trained model to ->", configs[model_type])
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


def main():
    train_model(model_type=args.model_type,
                bert_tf_hub=args.bert_tf_hub,
                train_data_path=args.train_loc,
                dev_data_path=args.dev_loc,
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
