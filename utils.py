import json
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import torch

label_encoding = {"entailment": 0, "contradiction": 1, "neutral": 2}


def load_nli_data(file_path):
    """
    Load Natural Language Inference (NLI) data from a JSON file and format it into a pandas DataFrame.

    This function reads each line of the file as a JSON object, creates a DataFrame,
    filters out examples with '-' gold labels, maps text labels to integers
    (entailment: 0, contradiction: 1, neutral: 2), and retains only the relevant columns.

    Parameters
    ----------
    file_path : str
        The path to the JSON file containing NLI data

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the processed NLI data with columns:
        - sentence1: The premise text
        - sentence2: The hypothesis text
        - label: Integer label (0: entailment, 1: contradiction, 2: neutral)
        - gold_label: Original string label
    """

    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    # Filter out examples with '-' gold label
    df = df[df["gold_label"] != "-"]

    # Map labels to integers
    df["label"] = df["gold_label"].map(label_encoding)

    # Keep only relevant columns
    df = df[["sentence1", "sentence2", "label", "gold_label"]]

    return df


def anli_to_snli(nli_set_path):
    """
    Convert the ANLI dataset to SNLI format.

    Reads JSON-lines from nli_set_path, maps ANLI label codes
    ('e','c','n') to SNLI string labels and integer encodings,
    writes the result to disk, and returns a DataFrame.

    References
    ----------
    https://github.com/facebookresearch/anli

    Parameters
    ----------
    nli_set_path : str
        Path to the ANLI JSONL file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
          - sentence1: premise text
          - sentence2: hypothesis text
          - label: integer label (0: entailment, 1: contradiction, 2: neutral)
          - gold_label: string label
    """
    total_data = []

    with open(nli_set_path, "r") as file_:
        for line in file_:
            data = {}
            eg = json.loads(line)
            data["sentence1"] = str(eg["context"])
            data["sentence2"] = str(eg["hypothesis"])
            # map to SNLI string labels
            if eg["label"] == "n":
                data["gold_label"] = "neutral"
            elif eg["label"] == "c":
                data["gold_label"] = "contradiction"
            elif eg["label"] == "e":
                data["gold_label"] = "entailment"
            # encode integer label
            data["label"] = label_encoding[data["gold_label"]]
            total_data.append(data)

    # create DataFrame
    df = pd.DataFrame(total_data)
    df = df[["sentence1", "sentence2", "label", "gold_label"]]

    return df


def tokenize_to_ids(texts, nlp, max_length=64, nr_unk=100):
    """
    Tokenize a list of texts into integer IDs using spaCy.

    This function converts texts into fixed-length sequences of token IDs that can be used for model input.
    Each token is either mapped to its corresponding row index in the embedding matrix (if it has a vector)
    or to a hash-based ID in a reserved range for unknown tokens.

    Parameters
    ----------
    texts : list of str
        List of text strings to tokenize.

    nlp : spacy.Language
        Loaded spaCy language model with word vectors.

    max_length : int, default=64
        Maximum length of each tokenized sequence. Longer sequences are truncated
        and shorter ones are padded with zeros.

    nr_unk : int, default=100
        Number of reserved IDs for unknown tokens. Unknown tokens are assigned
        IDs from 1 to nr_unk based on a hash of their text.

    Returns
    -------
    numpy.ndarray
        Array of shape (len(texts), max_length) containing token IDs for each text.
        Known tokens have IDs >= nr_unk + 1, unknown tokens have IDs between 1 and nr_unk,
        and padding has ID 0.

    Notes
    -----
    - Token IDs are assigned as follows:
      - 0: padding token
      - 1 to nr_unk: unknown tokens (based on hash)
      - nr_unk+1 and above: known tokens with vectors
    """

    vec_key2row = nlp.vocab.vectors.key2row  # dict: lex_id -> row_index
    all_ids = np.zeros((len(texts), max_length), dtype=np.int32)

    for i, doc in enumerate(
        tqdm(nlp.pipe(texts, n_process=-1), total=len(texts), desc="Tokenizing")
    ):
        seq_ids = []
        for token in doc[:max_length]:
            row = vec_key2row.get(token.orth)
            if row is not None and token.vector_norm > 0:
                seq_ids.append(row + nr_unk + 1)
            else:
                seq_ids.append(1 + (hash(token.text) % nr_unk))

        # pad/truncate
        arr = np.zeros(max_length, dtype=np.int32)
        arr[: len(seq_ids)] = seq_ids
        all_ids[i] = arr

    return all_ids


def get_embeddings_spacy(nlp, nr_unk=100):
    """
    Extract word embeddings from a spaCy language model and initialize the embedding matrix.

    Parameters
    ----------
    nlp : spacy.language.Language
        A loaded spaCy language model containing word vectors
    nr_unk : int, default=100
        Number of random vectors to generate for out-of-vocabulary (OOV) words

    Returns
    -------
    numpy.ndarray
        A matrix of shape (vocabulary_size + nr_unk + 1, embedding_dim) containing:
        - Index 0: reserved (zeros)
        - Indices 1 to nr_unk: random unit-norm vectors for OOV words
        - Remaining indices: word vectors from the spaCy model's vocabulary

    Notes
    -----
    The embedding matrix is initialized with zeros, then populated with random
    unit-normalized vectors for OOV tokens, followed by the pre-trained word
    vectors from the spaCy model.
    """

    vecs = nlp.vocab.vectors
    n_rows, dim = vecs.shape
    total = 1 + nr_unk + n_rows

    # init
    emb = np.zeros((total, dim), dtype="float32")

    # random OOV vectors (unit‑norm)
    oov = np.random.normal(size=(nr_unk, dim)).astype("float32")
    oov /= np.linalg.norm(oov, axis=1, keepdims=True)
    emb[1 : nr_unk + 1] = oov

    # copy spaCy vectors
    for lex_id, row in vecs.key2row.items():
        emb[nr_unk + 1 + row] = vecs.data[row]

    return emb


def compute_lengths(x):
    # count non-zero tokens per row
    return (x != 0).sum(dim=1)


def evaluate(model, loader, crit, device, return_loss=False):
    """
    Evaluate a model's performance on a dataset.

    This function runs the model in evaluation mode and computes accuracy and,
    optionally, the loss on a dataset. It processes batches from the data loader,
    computes sequence lengths for premises and hypotheses, and tracks correct
    predictions.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to evaluate
    loader : torch.utils.data.DataLoader
        DataLoader providing batches of data in the format (premise, hypothesis, label)
    crit : torch.nn.Module
        Loss criterion (e.g., CrossEntropyLoss) for computing the loss if return_loss is True
    device : torch.device
        Device to run the evaluation on (e.g., 'cuda' or 'cpu')
    return_loss : bool, default=False
        If True, returns both the average loss and accuracy.
        If False, returns only the accuracy.

    Returns
    -------
    float or tuple
        If return_loss is True, returns a tuple (avg_loss, accuracy).
        If return_loss is False, returns only the accuracy.
        Accuracy is the proportion of correctly classified samples.
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            l1, l2 = compute_lengths(x1), compute_lengths(x2)
            logits = model(x1, l1, x2, l2)
            preds = logits.argmax(1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
            if return_loss:
                loss = crit(logits, y)
                total_loss += loss.item() * y.size(0)

    accuracy = total_correct / total_samples
    if return_loss:
        return total_loss / total_samples, accuracy
    return accuracy
