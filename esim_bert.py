from keras import backend as K
from keras import layers, Model, models, optimizers, regularizers
from keras.layers import CuDNNLSTM
from keras.layers.wrappers import Bidirectional


def decomp_attention_model(vectors, shape, settings, embedding_type):
    max_length, nr_hidden, nr_class = shape
    input1, input2, x1, x2 = None, None, None, None

    input1 = layers.Input(shape=(1,), dtype="string", name="sent1")
    input2 = layers.Input(shape=(1,), dtype="string", name="sent2")

    embed = word_embedding_layer(vectors, max_length)

    x1 = embed(input1)
    x2 = embed(input2)

    # step 1: attend
    F = create_feedforward(num_units=nr_hidden)
    att_weights = layers.dot([F(x1), F(x2)], axes=-1)

    G = create_feedforward(num_units=nr_hidden)

    norm_weights_a = layers.Lambda(normalizer(1), name='attention_softmax_e1')(att_weights)
    norm_weights_b = layers.Lambda(normalizer(2), name='attention_softmax_e2')(att_weights)
    alpha = layers.dot([norm_weights_a, x1], axes=1, name='self_attend_a')
    beta = layers.dot([norm_weights_b, x2], axes=1, name='self_attend_b')

    # step 2: compare
    comp1 = layers.concatenate([x1, beta])
    comp2 = layers.concatenate([x2, alpha])
    v1 = layers.TimeDistributed(G)(comp1)
    v2 = layers.TimeDistributed(G)(comp2)

    # step 3: aggregate
    v1_sum = layers.Lambda(sum_word, name='sum_v1')(v1)
    v2_sum = layers.Lambda(sum_word, name='sum_v2')(v2)
    concat = layers.concatenate([v1_sum, v2_sum])

    H = create_feedforward(num_units=nr_hidden)
    out = H(concat)
    out = layers.Dense(nr_class, activation="softmax")(out)

    model = Model([input1, input2], out)

    model.compile(
        optimizer=optimizers.Adam(lr=settings["lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def word_embedding_layer(vectors, max_length):
    return models.Sequential(
        [
            layers.Embedding(
                vectors.shape[0],
                vectors.shape[1],
                input_length=max_length,
                weights=[vectors],
                trainable=False,
            )
        ]
    )


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
