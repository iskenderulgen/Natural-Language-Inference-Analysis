from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model, models, optimizers


def decomposable_attention_model(vectors, max_length, nr_hidden, nr_class, learning_rate):
    input1 = layers.Input(shape=(max_length,), dtype="int32", name="words1")
    input2 = layers.Input(shape=(max_length,), dtype="int32", name="words2")

    embed = word_embedding_layer(vectors=vectors, max_length=max_length, nr_hidden=nr_hidden)

    x1 = embed(input1)
    x2 = embed(input2)

    # step 1: attend
    F = create_feedforward(num_units=nr_hidden)
    att_weights = layers.dot([F(x1), F(x2)], axes=-1)

    #entailment direction is both sides.
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
    out = layers.Dense(nr_class, activation="softmax", name='last_classifier_layer')(out)

    model = Model([input1, input2], out)

    model.summary()

    model.compile(
        optimizer=optimizers.Adam(lr=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def word_embedding_layer(vectors, max_length, nr_hidden):
    return models.Sequential(
        [
            layers.Embedding(
                vectors.shape[0],
                vectors.shape[1],
                input_length=max_length,
                weights=[vectors],
                trainable=False,
            ),
            layers.TimeDistributed(
                layers.Dense(nr_hidden, activation=None, use_bias=False)
            ),
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


__all__ = [decomposable_attention_model]
