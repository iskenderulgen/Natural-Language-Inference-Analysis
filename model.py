from keras import backend as K
from keras import layers, Model, models, optimizers
from keras.layers import CuDNNLSTM
from keras.layers.wrappers import Bidirectional


def esim_bilstm_model(vectors, shape, settings):
    max_length, nr_hidden, nr_class = shape

    bilstm1 = Bidirectional(CuDNNLSTM(nr_hidden, return_sequences=True))
    bilstm2 = Bidirectional(CuDNNLSTM(nr_hidden, return_sequences=True))

    i1 = layers.Input(shape=(max_length,), dtype="int32", name="words1")
    print("input 1 is :", K.shape(i1))
    i2 = layers.Input(shape=(max_length,), dtype="int32", name="words2")
    print("input 2 is :", K.shape(i2))

    embed = word_embedding_layer(vectors, max_length, nr_hidden)

    x1 = embed(i1)
    x2 = embed(i2)

    x1 = bilstm1(x1)
    x2 = bilstm1(x2)

    e = layers.Dot(axes=2, name="attention_dot_product")([x1, x2])
    attend_e1 = layers.Softmax(axis=2, name="attention_softmax_e1")(e)
    attend_e2 = layers.Softmax(axis=1, name="attention_softmax_e2")(e)
    e1 = layers.Lambda(K.expand_dims, arguments={'axis': 3}, name="expand_dim_e1")(attend_e1)
    e2 = layers.Lambda(K.expand_dims, arguments={'axis': 3}, name="expand_dim_e2")(attend_e2)

    _x1 = layers.Lambda(K.expand_dims, arguments={'axis': 1}, name="second_expand_dims_x1")(x2)
    _x1 = layers.Multiply(name="multiply_x1")([e1, _x1])
    _x1 = layers.Lambda(K.sum, arguments={'axis': 2}, name="sum_x1")(_x1)
    _x2 = layers.Lambda(K.expand_dims, arguments={'axis': 2}, name="second_expand_dims_x2")(x1)
    _x2 = layers.Multiply(name="multiply_x2")([e2, _x2])
    _x2 = layers.Lambda(K.sum, arguments={'axis': 1}, name="sum_x2")(_x2)

    m1 = layers.Concatenate(name="Concatenate_m1")(
        [x1, _x1, layers.Subtract()([x1, _x1]), layers.Multiply()([x1, _x1])])
    m2 = layers.Concatenate(name="Concatenate_m2")(
        [x2, _x2, layers.Subtract()([x2, _x2]), layers.Multiply()([x2, _x2])])

    y1 = bilstm2(m1)
    y2 = bilstm2(m2)

    mx1 = layers.Lambda(K.max, arguments={'axis': 1}, name="mx1_maxiumum")(y1)
    av1 = layers.Lambda(K.mean, arguments={'axis': 1}, name="av1_mean")(y1)
    mx2 = layers.Lambda(K.max, arguments={'axis': 1}, name="mx2_maximum")(y2)
    av2 = layers.Lambda(K.mean, arguments={'axis': 1}, name="av2_mean")(y2)

    y = layers.Concatenate(name="Last_big_concat")([av1, mx1, av2, mx2])
    y = layers.Dense(1024)(y)
    y = layers.LeakyReLU(alpha=0.3)(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(1024)(y)
    y = layers.LeakyReLU(alpha=0.3)(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(3, activation='softmax')(y)

    model = Model(inputs=[i1, i2], outputs=[y])

    model.compile(
        optimizer=optimizers.Adam(lr=settings["lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def decomp_attention_model(vectors, shape, settings, train_type):
    # shape = 64 , 200 , 3
    max_length, nr_hidden, nr_class = shape
    input1, input2, a, b = None, None, None, None
    if train_type == "word":
        input1 = layers.Input(shape=(max_length,), dtype="int32", name="words1")
        print("input 1 is :", K.shape(input1))
        input2 = layers.Input(shape=(max_length,), dtype="int32", name="words2")
        print("input 2 is :", K.shape(input2))

        # embeddings (projected)
        # vector shape is [684,825 , 300] -> for spacy
        # vector shape is [35,622 , 1024] -> for bert
        embed = word_embedding_layer(vectors, max_length, nr_hidden)

        a = embed(input1)
        print("a is :", K.shape(a))
        b = embed(input2)
        print("b is :", K.shape(b))

    elif train_type == "sentence":

        input1 = layers.Input(shape=(max_length,), dtype="float32", name="sentence1")
        print("input 1 shape is :", K.shape(input1))

        input2 = layers.Input(shape=(max_length,), dtype="float32", name="sentence2")
        print("input 2 shape is :", K.shape(input2))

        bert = sentence_embedding_layer(projected_dim=nr_hidden)

        a = bert(input1)
        print("a shape is :", K.shape(a))

        b = bert(input2)
        print("b shape is :", K.shape(b))

    else:
        print("unknown model type")

    # step 1: attend
    F = create_feedforward(num_units=nr_hidden)
    att_weights = layers.dot([F(a), F(b)], axes=-1)

    G = create_feedforward(num_units=nr_hidden)

    norm_weights_a = layers.Lambda(normalizer(1), name='attention_softmax_e1')(att_weights)
    norm_weights_b = layers.Lambda(normalizer(2), name='attention_softmax_e2')(att_weights)
    alpha = layers.dot([norm_weights_a, a], axes=1, name='self_attend_a')
    beta = layers.dot([norm_weights_b, b], axes=1, name='self_attend_b')

    # step 2: compare
    comp1 = layers.concatenate([a, beta])
    comp2 = layers.concatenate([b, alpha])
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


def sentence_embedding_layer(projected_dim):
    return models.Sequential(
        [
            layers.Reshape(
                (1, 1024), input_shape=(1024,)
            ),
            layers.TimeDistributed(
                layers.Dense(projected_dim, activation=None, use_bias=False),
                input_shape=(1024,)
            ),
        ]
    )


def word_embedding_layer(vectors, max_length, projected_dim):
    return models.Sequential(
        [
            layers.Embedding(
                # vocab / vector size - total word vector size
                vectors.shape[0],
                # dimension of word vectors (300 or 1024)
                vectors.shape[1],
                # max sequence length (64 words max)
                input_length=max_length,
                weights=[vectors],
                trainable=False,
            ),
            layers.TimeDistributed(
                # 200 dim projected layer
                # turns (?,64,300) layer to (?,64,200)
                layers.Dense(projected_dim, activation=None, use_bias=False)
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


__all__ = [decomp_attention_model]
