from keras import backend as K
from keras import layers, Model, optimizers, models
from keras.layers import CuDNNLSTM
from keras.layers.wrappers import Bidirectional


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

    e = layers.Dot(axes=2, name="dot product of x1 x2")([x1, x2])
    attend_e1 = layers.Softmax(axis=2, name="softmax of dot on axis 2")(e)
    attend_e2 = layers.Softmax(axis=1, name="softmax of dot on axis 1")(e)
    e1 = layers.Lambda(K.expand_dims, arguments={'axis': 3}, name="expand dim on attend_e1")(attend_e1)
    e2 = layers.Lambda(K.expand_dims, arguments={'axis': 3}, name="expand dim on attend_e2")(attend_e2)

    _x1 = layers.Lambda(K.expand_dims, arguments={'axis': 1}, name="expand dim on x2")(x2)
    _x1 = layers.Multiply(name="multiply e1 with expanded x2")([e1, _x1])
    _x1 = layers.Lambda(K.sum, arguments={'axis': 2}, name="sum_x1")(_x1)
    _x2 = layers.Lambda(K.expand_dims, arguments={'axis': 2}, name="expand dim on x1")(x1)
    _x2 = layers.Multiply(name="multiply e2 with expanded x1")([e2, _x2])
    _x2 = layers.Lambda(K.sum, arguments={'axis': 1}, name="sum_x2")(_x2)

    m1 = layers.Concatenate(name="concatenate x1 with attended x1")(
        [x1, _x1, layers.Subtract()([x1, _x1]), layers.Multiply()([x1, _x1])])
    m2 = layers.Concatenate(name="concatenate x2 with attended x2")(
        [x2, _x2, layers.Subtract()([x2, _x2]), layers.Multiply()([x2, _x2])])

    y1 = bilstm2(m1)
    y2 = bilstm2(m2)

    mx1 = layers.Lambda(K.max, arguments={'axis': 1}, name="k.max of y1")(y1)
    av1 = layers.Lambda(K.mean, arguments={'axis': 1}, name="k.mean of y1")(y1)
    mx2 = layers.Lambda(K.max, arguments={'axis': 1}, name="k.max of y2")(y2)
    av2 = layers.Lambda(K.mean, arguments={'axis': 1}, name="k.mean of y2")(y2)

    y = layers.Concatenate(name="concatenate max and mean of y1 and y2")([av1, mx1, av2, mx2])
    y = layers.Dense(1024)(y)
    y = layers.ReLU()(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(512)(y)
    y = layers.ReLU()(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(nr_class, activation='last classifier layer')(y)

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
