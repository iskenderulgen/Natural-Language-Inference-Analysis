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
