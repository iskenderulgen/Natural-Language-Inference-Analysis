import numpy as np
from keras import Model
from keras.models import load_model

try:
    import cPickle as pickle
except ImportError:
    import pickle
import tensorflow as tf
from bert_dependencies import tokenization


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


class SpacyPrediction(object):
    entailment_types = ["entailment", "contradiction", "neutral"]

    @classmethod
    def load(cls, path, max_length=100, get_features=None):
        if get_features is None:
            get_features = get_word_ids

        model = load_model(path, custom_objects={"tf": tf})
        print("loading model")
        #############
        model = Model(inputs=model.input,
                      outputs=[model.output,
                               model.get_layer('sum_x1').output,
                               model.get_layer('sum_x2').output])
        #############
        model.summary()
        print("Loaded model from disk")

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
        outputs = self.model.predict([x1, x2])
        scores = outputs[0]

        return self.entailment_types[scores.argmax()], scores.max(), outputs[1], outputs[2]


class BertWordPredict(object):
    entailment_types = ["entailment", "contradiction", "neutral"]

    @staticmethod
    def read_examples(input_sentences):
        """Read a list of `InputExample`s from an input file."""

        if type(input_sentences) is np.ndarray or list:
            print("input file is list / array")
            examples = []
            total_sentences_count = len(input_sentences)
            print("Total sentences to be read: ", total_sentences_count)

            for sentence in input_sentences:
                line = tokenization.convert_to_unicode(sentence).strip()
                examples.append(line)
            return examples

        else:
            return TypeError

    @staticmethod
    def convert_examples_to_features(examples, seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        # converts sentences in to features such as id's of the tokens, not full vector representations.
        features = []
        sentence_tokens = []
        for (sent_index, sentence) in enumerate(examples):
            tokens = tokenizer.tokenize(sentence)

            if len(tokens) > seq_length:
                tokens = tokens[0:seq_length]

            # This is the part that tokens converted to ids.
            input_ids_raw = tokenizer.convert_tokens_to_ids(tokens)

            # Zero-pad up to the sequence length.
            while len(input_ids_raw) < seq_length:
                input_ids_raw.append(0)
            assert len(input_ids_raw) == seq_length

            features.append(input_ids_raw)
            sentence_tokens.append(tokens)

        return features, sentence_tokens

    @staticmethod
    def predict(premises, hypothesis, path, transformer_type, eval_type, label):
        entailment_types = ["entailment", "contradiction", "neutral"]

        model = load_model(path + 'similarity/' + transformer_type + "_" + "model.h5"
                           , custom_objects={"tf": tf})
        print("loading model")
        model.summary()
        model = Model(inputs=model.input,
                      outputs=[model.output,
                               model.get_layer('sum_x1').output,
                               model.get_layer('sum_x2').output])

        print("Loaded model from disk")
        tokenizer = tokenization.FullTokenizer(
            vocab_file=path + "transformers/bert/vocab.txt", do_lower_case=True)

        if eval_type == 'demo':
            sentences = [premises, hypothesis]
            examples = BertWordPredict.read_examples(sentences)
            sentences_features, sentence_tokens = BertWordPredict.convert_examples_to_features(examples=examples,
                                                                                               seq_length=64,
                                                                                               tokenizer=tokenizer)
            premise_vectors = np.asarray(sentences_features[0]).reshape((1, 64))
            hypothesis_vectors = np.asarray(sentences_features[1]).reshape((1, 64))
            outputs = model.predict([premise_vectors, hypothesis_vectors])

            scores = outputs[0]
            return entailment_types[scores.argmax()], scores.max(), outputs[1], outputs[2], sentence_tokens

        elif eval_type == 'demo_listlike':
            premises = BertWordPredict.read_examples(premises)
            hypothesis = BertWordPredict.read_examples(hypothesis)
            premise_features, _ = BertWordPredict.convert_examples_to_features(examples=premises,
                                                                               seq_length=64,
                                                                               tokenizer=tokenizer)
            hypothesis_features, _ = BertWordPredict.convert_examples_to_features(examples=hypothesis,
                                                                                  seq_length=64,
                                                                                  tokenizer=tokenizer)
            total = 0.0
            contradict = 0.0
            entailment = 0.0
            neutral = 0.0
            for text1, text2 in zip(premise_features, hypothesis_features):
                premise_vectors = np.asarray(text1).reshape((1, 64))
                hypothesis_vectors = np.asarray(text2).reshape((1, 64))
                outputs = model.predict([premise_vectors, hypothesis_vectors])
                if entailment_types[outputs[0].argmax()] is 'contradiction':
                    contradict += 1
                elif entailment_types[outputs[0].argmax()] is 'entailment':
                    entailment += 1
                elif entailment_types[outputs[0].argmax()] is 'neutral':
                    neutral += 1
                total += 1

            print("total contradiction = ", contradict / total)
            print("total entailment =", entailment / total)
            print("total neutral =", neutral / total)

        elif eval_type == 'evaluate':
            total = 0.0
            true_p = 0.0

            premises = BertWordPredict.read_examples(premises)
            hypothesis = BertWordPredict.read_examples(hypothesis)
            premise_features, _ = BertWordPredict.convert_examples_to_features(examples=premises,
                                                                               seq_length=64,
                                                                               tokenizer=tokenizer)
            hypothesis_features, _ = BertWordPredict.convert_examples_to_features(examples=hypothesis,
                                                                                  seq_length=64,
                                                                                  tokenizer=tokenizer)

            for i in range(len(premise_features)):
                premise_vectors = np.asarray(premise_features[i]).reshape((1, 64))
                hypothesis_vectors = np.asarray(hypothesis_features[i]).reshape((1, 64))
                outputs = model.predict([premise_vectors, hypothesis_vectors])
                scores = outputs[0]
                # if entailment_types[scores.argmax()] == 'contradiction':
                #     y_pred = 'neutral'
                # else:
                #     y_pred = entailment_types[scores.argmax()]
                if entailment_types[scores.argmax()] == BertWordPredict.entailment_types[label[i].argmax()]:
                    true_p += 1
                total += 1
            print("Entailment Model Accuracy is", true_p / total)
