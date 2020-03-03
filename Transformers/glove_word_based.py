import datetime


def create_dataset_ids(nlp, texts, hypotheses, num_unk, max_length):
    """This section creates id matrix of the input tokens"""

    sents = texts + hypotheses
    sents_as_ids = []

    print("Total number of sentences to be processed = ", len(sents))
    starttime = datetime.datetime.now()
    count = 0