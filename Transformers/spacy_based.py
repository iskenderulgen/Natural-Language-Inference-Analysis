import datetime
import os
import pickle
import cupy as cp
import numpy as np

from Transformers.utils import read_snli, load_spacy_nlp


def create_dataset_ids(nlp, texts, hypotheses, num_unk, max_length):
    """This section creates id matrix of the input tokens"""

    sents = texts + hypotheses
    sents_as_ids = []

    print("Total number of sentences to be processed = ", len(sents))
    starttime = datetime.datetime.now()
    count = 0

    for sent in sents:
        doc = nlp(sent, disable=['parser', 'tagger', 'ner', 'textcat'])
        word_ids = []
        for i, token in enumerate(doc):
            # i indisi word leri tek tek numaralandırıyor bu sayede max lenght ile karsılastırır.
            # skip odd spaces from tokenizer
            if token.has_vector and token.vector_norm == 0:
                continue
            if i > max_length:
                break
            if token.has_vector:
                word_ids.append(token.rank + num_unk + 1)
            else:
                # if we don't have a vector, pick an OOV entry
                word_ids.append(token.rank % num_unk + 1)

        # there must be a simpler way of generating padded arrays from lists...
        word_id_vec = np.zeros(max_length, dtype="int")
        clipped_len = min(max_length, len(word_ids))
        word_id_vec[:clipped_len] = word_ids[:clipped_len]
        sents_as_ids.append(word_id_vec)

        count = count + 1
        if count % 50000 == 0:
            print("total sentence: " + str(count) + " Total percent: " + str(count / len(sents)))

    finishtime = datetime.datetime.now()
    totaltime = finishtime - starttime

    print("Total time elapse:" + str(totaltime))
    # text ler ve hipotezleri ayrı ayrı diziler olarak alıyor birinci kısım text - ikinci kısım hipotez
    return [np.array(sents_as_ids[: len(texts)]), np.array(sents_as_ids[len(texts):])]


def get_embeddings(vocab, nr_unk=100):
    # the extra +1 is for a zero vector representing sentence-final padding
    num_vectors = max(lex.rank for lex in vocab) + 2

    # create random vectors for OOV tokens
    oov = np.random.normal(size=(nr_unk, vocab.vectors_length))
    oov = oov / oov.sum(axis=1, keepdims=True)

    vectors = np.zeros((num_vectors + nr_unk, vocab.vectors_length), dtype="float32")
    vectors[1: (nr_unk + 1), ] = oov
    for lex in vocab:
        if lex.has_vector and lex.vector_norm > 0:
            # burada vector olan yerler Cupy dizisi bunu numpy a cevirmek gerekiyormus
            vectors[nr_unk + 1 + lex.rank] = cp.asnumpy(lex.vector / lex.vector_norm)
    # vector shape is [684,925 , 300]
    print("getting embeddings from spacy vocabulary is finished")

    return vectors


def spacy_word_transformer(path, train_loc, dev_loc, shape, transformer_type):
    print("Transformer type is ", transformer_type)

    nlp = load_spacy_nlp()

    train_texts1, train_texts2, train_labels = read_snli(train_loc)
    dev_texts1, dev_texts2, dev_labels = read_snli(dev_loc)
    print("Processing texts using spacy")

    if os.path.isfile(path=path + "Processed_SNLI/Spacy_Processed/train_x.pkl"):
        print("Spacy based Pre-Processed train file is found now loading")
        with open(path + 'Processed_SNLI/Spacy_Processed/train_x.pkl', 'rb') as f:
            train_x = pickle.load(f)
    else:
        print("There is no spacy based pre-processed file of train_X, Pre-Process will start now")
        train_x = create_dataset_ids(nlp=nlp, texts=train_texts1, hypotheses=train_texts2, num_unk=100,
                                     max_length=shape[0])
        with open(path + 'Processed_SNLI/Spacy_Processed/train_x.pkl', 'wb') as f:
            pickle.dump(train_x, f)

    if os.path.isfile(path=path + "Processed_SNLI/Spacy_Processed/dev_x.pkl"):
        print("Spacy based Pre-Processed dev file is found now loading")
        with open(path + 'Processed_SNLI/Spacy_Processed/dev_x.pkl', 'rb') as f:
            dev_x = pickle.load(f)
    else:
        print("There is no spacy based pre-processed file of dev_X, Pre-Process will start now")
        dev_x = create_dataset_ids(nlp=nlp, texts=dev_texts1, hypotheses=dev_texts2, num_unk=100, max_length=shape[0])
        with open(path + 'Processed_SNLI/Spacy_Processed/dev_x.pkl', 'wb') as f:
            pickle.dump(dev_x, f)

    if os.path.isfile(path=path + "Processed_SNLI/Spacy_Processed/spacy_weights.pkl"):
        print("Spacy weights matrix already extracted, now loading...")
        with open(path + 'Processed_SNLI/Spacy_Processed/spacy_weights.pkl', 'rb') as f:
            vectors = pickle.load(f)
    else:
        print("Spacy weight matrix is not found, now extracting...")
        vectors = get_embeddings(vocab=nlp.vocab, nr_unk=100)
        with open(path + 'Processed_SNLI/Spacy_Processed/spacy_weights.pkl', 'wb') as f:
            pickle.dump(vectors, f)

    return train_x, train_labels, dev_x, dev_labels, vectors
