import datetime
import numpy as np

"""This section creates id matrix of the inout tokens"""


def create_dataset(nlp, texts, hypotheses, num_unk, max_length):
    sents = texts + hypotheses
    # print(sents)
    sents_as_ids = []

    print(len(sents))
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

def spacy_word_transformer():
    print("Processing texts using spacy")

    if os.path.isfile(path=path + "Processed_SNLI/Spacy_Processed/train_x.pkl"):
        print("Pre-Processed train file is found now loading")
        with open(path + 'Processed_SNLI/Spacy_Processed/train_x.pkl', 'rb') as f:
            train_x = pickle.load(f)
    else:
        print("There is no pre-processed file of train_X, Pre-Process will start now")
        train_x = create_dataset(nlp=nlp, texts=train_texts1, hypotheses=train_texts2, num_unk=100,
                                 max_length=shape[0])
        with open(path + 'Processed_SNLI/Spacy_Processed/train_x.pkl', 'wb') as f:
            pickle.dump(train_x, f)

    if os.path.isfile(path=path + "Processed_SNLI/Spacy_Processed/dev_x.pkl"):
        print("Pre-Processed dev file is found now loading")
        with open(path + 'Processed_SNLI/Spacy_Processed/dev_x.pkl', 'rb') as f:
            dev_x = pickle.load(f)
    else:
        print("There is no pre-processed file of dev_X, Pre-Process will start now")
        dev_x = create_dataset(nlp=nlp, texts=dev_texts1, hypotheses=dev_texts2, num_unk=100, max_length=shape[0])
        with open(path + 'Processed_SNLI/Spacy_Processed/dev_x.pkl', 'wb') as f:
            pickle.dump(dev_x, f)

    return train_x, train_labels, dev_x, dev_labels