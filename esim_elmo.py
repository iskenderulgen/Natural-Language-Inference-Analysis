# from keras import backend as K
# from keras import layers, Model, models, optimizers, regularizers
# from keras.layers import CuDNNLSTM, LSTM, Layer
# from keras.layers.wrappers import Bidirectional
# import tensorflow_hub as hub
# import tensorflow as tf
#
# def esim_bilstm_model(vectors, shape, settings, embedding_type):
#     max_length, nr_hidden, nr_class = shape
#     input1, input2, x1, x2 = None, None, None, None
#
#     bilstm1 = Bidirectional(CuDNNLSTM(nr_hidden, return_sequences=True))
#     bilstm2 = Bidirectional(CuDNNLSTM(nr_hidden, return_sequences=True))
#
#     input1 = layers.Input(shape=(max_length,), dtype="int32", name="words1")
#     input2 = layers.Input(shape=(max_length,), dtype="int32", name="words2")
#
#     x1 = bilstm1(x1)
#     x2 = bilstm1(x2)
#
#     e = layers.Dot(axes=2, name="attention_dot_product")([x1, x2])
#     attend_e1 = layers.Softmax(axis=2, name="attention_softmax_e1")(e)
#     attend_e2 = layers.Softmax(axis=1, name="attention_softmax_e2")(e)
#     e1 = layers.Lambda(K.expand_dims, arguments={'axis': 3}, name="expand_dim_e1")(attend_e1)
#     e2 = layers.Lambda(K.expand_dims, arguments={'axis': 3}, name="expand_dim_e2")(attend_e2)
#
#     _x1 = layers.Lambda(K.expand_dims, arguments={'axis': 1}, name="second_expand_dims_x1")(x2)
#     _x1 = layers.Multiply(name="multiply_x1")([e1, _x1])
#     _x1 = layers.Lambda(K.sum, arguments={'axis': 2}, name="sum_x1")(_x1)
#     _x2 = layers.Lambda(K.expand_dims, arguments={'axis': 2}, name="second_expand_dims_x2")(x1)
#     _x2 = layers.Multiply(name="multiply_x2")([e2, _x2])
#     _x2 = layers.Lambda(K.sum, arguments={'axis': 1}, name="sum_x2")(_x2)
#
#     m1 = layers.Concatenate(name="Concatenate_m1")(
#         [x1, _x1, layers.Subtract()([x1, _x1]), layers.Multiply()([x1, _x1])])
#     m2 = layers.Concatenate(name="Concatenate_m2")(
#         [x2, _x2, layers.Subtract()([x2, _x2]), layers.Multiply()([x2, _x2])])
#
#     y1 = bilstm2(m1)
#     y2 = bilstm2(m2)
#
#     mx1 = layers.Lambda(K.max, arguments={'axis': 1}, name="mx1_maxiumum")(y1)
#     av1 = layers.Lambda(K.mean, arguments={'axis': 1}, name="av1_mean")(y1)
#     mx2 = layers.Lambda(K.max, arguments={'axis': 1}, name="mx2_maximum")(y2)
#     av2 = layers.Lambda(K.mean, arguments={'axis': 1}, name="av2_mean")(y2)
#
#     y = layers.Concatenate(name="Last_big_concat")([av1, mx1, av2, mx2])
#     y = layers.Dense(1024)(y)
#     y = layers.LeakyReLU(alpha=0.3)(y)
#     y = layers.Dropout(0.2)(y)
#     y = layers.Dense(1024)(y)
#     y = layers.LeakyReLU(alpha=0.3)(y)
#     y = layers.Dropout(0.2)(y)
#     y = layers.Dense(3, activation='softmax')(y)
#
#     model = Model(inputs=[input1, input2], outputs=[y])
#
#     model.compile(
#         optimizer=optimizers.Adam(lr=settings["lr"]),
#         loss="categorical_crossentropy",
#         metrics=["accuracy"],
#     )
#
#     return model
#
#
# class ElmoEmbeddingLayer(Layer):
#     def __init__(self, trainable=True, **kwargs):
#         self.dimensions = 1024
#         self.trainable = trainable
#         super(ElmoEmbeddingLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.elmo = hub.Module('https://tfhub.dev/google/elmo/3', trainable=self.trainable,
#                                name="{}_module".format(self.name))
#         self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
#         super(ElmoEmbeddingLayer, self).build(input_shape)
#
#     def call(self, x, mask=None):
#         result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
#                            as_dict=True,
#                            signature='default',
#                            )['default']
#         return result
#
#     def compute_mask(self, inputs, mask=None):
#         return K.not_equal(inputs, '--PAD--')
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], self.dimensions
#
#
# def data_load():
#     sents = premises + hypothesis
#
#     sentence_tokens = []
#     tokens_length = []
#
#     for sent in sents:
#         tokens = []
#         doc = nlp(sent, disable=['parser', 'tagger', 'ner', 'textcat'])
#         for token in doc:
#             tokens.append(token.text)
#         tokens_length.append(len(doc))
#         sentence_tokens.append(tokens)
#
#     # max_length = np.amax(tokens_length)
#
#     for sent_token in sentence_tokens:
#         while len(sent_token) > 50:
#             sent_token.pop()
#         while len(sent_token) < 50:
#             sent_token.append("")