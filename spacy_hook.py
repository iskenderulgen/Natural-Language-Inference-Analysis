
import numpy as np
from keras.models import model_from_json

from Transformers.spacy_based import get_embeddings

try:
    import cPickle as pickle
except ImportError:
    import pickle


class KerasSimilarityShim(object):
    entailment_types = ["entailment", "contradiction", "neutral"]

    @classmethod
    def load(cls, path, nlp, max_length=100, get_features=None):
        if get_features is None:
            get_features = get_word_ids

        with (path / 'config.json').open() as file_:
            model = model_from_json(file_.read())
        with (path / 'model').open('rb') as file_:
            weights = pickle.load(file_)

        embeddings = get_embeddings(nlp.vocab)
        weights.insert(1, embeddings)
        model.set_weights(weights)

        return cls(model, get_features=get_features, max_length=max_length)

    def __init__(self, model, get_features=None, max_length=100):
        self.model = model
        self.get_features = get_features
        self.max_length = max_length

    def __call__(self, doc):
        doc.user_hooks["similarity"] = self.predict
        doc.user_span_hooks["similarity"] = self.predict

        return doc

    def predict(self, doc1, doc2):
        x1 = self.get_features([doc1], max_length=self.max_length)
        x2 = self.get_features([doc2], max_length=self.max_length)
        scores = self.model.predict([x1, x2])

        return self.entailment_types[scores.argmax()], scores.max()


def get_word_ids(docs, max_length=100, nr_unk=100):
    xs = np.zeros((len(docs), max_length), dtype="int32")

    for i, doc in enumerate(docs):
        for j, token in enumerate(doc):
            if j == max_length:
                break
            if token.has_vector:
                xs[i, j] = token.rank + nr_unk + 1
            else:
                xs[i, j] = token.rank % nr_unk + 1
    return xs
