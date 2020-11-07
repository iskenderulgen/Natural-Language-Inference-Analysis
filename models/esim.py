from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model, optimizers, models
from tensorflow.keras.layers import Bidirectional, CuDNNLSTM


def esim_bilstm_model(vectors, max_length, nr_hidden, nr_class, learning_rate, embedding_type):
    input1, input2, x1, x2 = None, None, None, None

    bilstm1 = Bidirectional(CuDNNLSTM(nr_hidden, return_sequences=True))
    bilstm2 = Bidirectional(CuDNNLSTM(nr_hidden, return_sequences=True))

    if embedding_type == 'word':
        input1 = layers.Input(shape=(max_length,), dtype="int32", name="words1")
        input2 = layers.Input(shape=(max_length,), dtype="int32", name="words2")

        embed = word_embedding_layer(vectors, max_length)

        x1 = embed(input1)
        x2 = embed(input2)

    elif embedding_type == 'sentence':
        input1 = layers.Input(shape=(max_length,), dtype="float32", name="sentence1")
        input2 = layers.Input(shape=(max_length,), dtype="float32", name="sentence2")

        embed = sentence_embedding_layer()

        x1 = embed(input1)
        x2 = embed(input2)

    else:
        print("unknown embedding type, Embedding type can only be 'word' or 'sentence' ")

    x1 = bilstm1(x1)
    x2 = bilstm1(x2)

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

    model = Model(inputs=[input1, input2], outputs=[y])

    model.compile(
        optimizer=optimizers.Adam(lr=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def sentence_embedding_layer():
    return models.Sequential(
        [
            layers.Reshape(
                (1, 1024), input_shape=(1024,)
            ),
        ]
    )


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
