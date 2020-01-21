# This file contains Exploratory Data Analysis Results

SNLI corpus is hand crafted entailment - contradiction corpus that created by stanford university researchers. Raw SNLI corpus is human-labeled using 5 annotators as (Contradiction - Entailment - Neutral). The gold label is the mean of the human labels.
Standart SNLI corpus data type is JSONL and contains following features.

### SNLI Train JSONL
```
* SNLI_train.JSONL ->  is 549367 rows combined of sentence pairs (it has sentece - gold label - labels and POS tags)
* Raw json file is 487 MB of data
* File size after using BERT transformer is 6GB (reason of the huge file size is that bert used 15kb of data per token)
```

### SNLI Dev JSONL
```
* SNLI_dev.JSONL ->  is 9842 rows combined of sentence pairs (it has sentece - gold label - labels and POS tags)
* Raw json file is 9 MB of data
* File size after using BERT transformer is 120MB (reason of the huge file size is that bert used 15kb of data per token)
```
### SNLI Test JSONL
```
* SNLI_dev.JSONL ->  is 9842 rows combined of sentence pairs (it has sentece - gold label - labels and POS tags)
* Raw json file is 9 MB of data
* File size after using BERT transformer is 120MB (reason of the huge file size is that bert used 15kb of data per token)
```

SNLI dev and test data has the same row size. The raw file is fixed as 550,000 train - 10,000 Dev / Test. But some paris has no gold label and for the sake of the NN model we exclude these pairs.

