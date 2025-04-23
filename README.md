# Natural Language Inference (NLI) Text Classification

## What is Natural Language Inference?

Natural Language Inference (NLI) is a fundamental NLP task that involves determining the logical relationship between a premise (a given statement) and a hypothesis (a proposed statement). The possible relationships are typically:

- **Entailment**: The hypothesis logically follows from the premise
- **Contradiction**: The hypothesis contradicts the premise
- **Neutral**: The hypothesis neither follows from nor contradicts the premise

### NLI Datasets and Benchmarks:

- **SNLI** (Stanford Natural Language Inference): [https://nlp.stanford.edu/projects/snli/](https://nlp.stanford.edu/projects/snli/) - 570K human-written sentence pairs with balanced labels
- **MultiNLI** (Multi-Genre NLI): [https://cims.nyu.edu/~sbowman/multinli/](https://cims.nyu.edu/~sbowman/multinli/) - 433K sentence pairs from diverse text genres
- **Adversarial NLI**: [https://github.com/facebookresearch/anli](https://github.com/facebookresearch/anli) - Dynamically collected through human-and-model-in-the-loop process to challenge state-of-the-art models

This repository provides tools and scripts for exploring and training Natural Language Inference (NLI) models, with a focus on contradiction detection. It leverages spaCy embeddings, a custom ESIM implementation, and Hugging Face Transformers for BERT-based experiments.

## Repository Structure

```
Contradiction/
├── utils/
│   ├── utils.py                      # Common utilities and helper functions
│   └── exploratory_analysis.ipynb    # Data exploration notebook
├── models/
│   └── esim.py                       # ESIM model implementation
├── nli_train_spacy.ipynb             # Training pipeline with spaCy embeddings
├── nli_train_bert.ipynb              # Fine-tuning BERT for NLI
├── nli_prediction.ipynb              # Interactive inference & visualization
├── data/                             # Preprocessed data (git-ignored)
├── LICENSE                           # Apache 2.0 License
└── README.md                         # Project documentation
```

- **utils/**
  - `utils.py`  
    Common data-loading, tokenization, embedding extraction, evaluation, and helper functions.
  - `exploratory_analysis.ipynb`  
    Notebook for in-depth data exploration: token counts, similarity analysis, lexical overlap, and t‑SNE visualization.

- **models/**
  - `esim.py`  
    Custom PyTorch implementation of the ESIM architecture (BiLSTM + attention + pooling). [https://arxiv.org/abs/1609.06038]

- **nli_train_spacy.ipynb**  
  End-to-end pipeline: preprocess SNLI data, generate token IDs & embedding matrix, train and evaluate ESIM with spaCy embeddings.

- **nli_train_bert.ipynb**  
  BERT-based classification on SNLI using Hugging Face Trainer API and custom metric integration.

- **nli_prediction.ipynb**  
  Interactive inference notebook:
  - ESIM attention visualization (premise↔hypothesis)
  - BERT attention extraction and heatmap plotting

## Key Components

1. **Data Loading & Preprocessing**  
   - JSONL reader for SNLI & ANLI formats  
   - spaCy‑based tokenization → fixed‑length ID sequences  
   - Embedding matrix construction with random OOV vectors

2. **Exploratory Analysis**  
   - Token count statistics per label  
   - Cosine similarity distributions  
   - Jaccard lexical overlap  
   - t‑SNE projection of composite sentence‑pair embeddings

3. **Modeling**  
   - **ESIM**: two-stage BiLSTM with soft‑attention and pooling  
   - **BERT**: fine‑tuning BERT for sequence classification via Transformers

4. **Evaluation & Inference**  
   - Accuracy & loss computation  
   - Interactive visualization of Attention heatmaps for interpretability  
   - Pipeline for both spaCy‑based and Transformer‑based models

## Prerequisites

```
- cupy-cuda12x==12.3.0  
- en_core_web_lg==3.8.0  
- evaluate==0.4.3  
- scikit-learn==1.6.1  
- seaborn==0.13.2  
- spacy==3.8.5  
- torch==2.6.0+cu126  
- numpy==2.2.5  
- pandas==2.2.3  
- matplotlib==3.10  
- transformers==4.51.3  
- datasets==3.5.0  
- tqdm==4.67.1
```

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
## Acknowledgments
- [Stanford NLI](https://nlp.stanford.edu/projects/snli/)
- [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/)
- [Adversarial NLI](https://github.com/facebookresearch/anli)
- [ESIM](https://arxiv.org/abs/1609.06038)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [spaCy](https://spacy.io/)
- [PyTorch](https://pytorch.org/)
- [BERT](https://arxiv.org/abs/1810.04805)

# References


```
@inproceedings{9478044,
  author={Oğul, İskender Ülgen and Tekır, Selma},
  booktitle={2021 29th Signal Processing and Communications Applications Conference (SIU)}, 
  title={Performance Evaluation of BERT Vectors on Natural Language Inference Models}, 
  year={2021},
  volume={},
  number={},
  pages={1-4},
  doi={10.1109/SIU53274.2021.9478044}}
```

```
@mastersthesis{ogul2020classification,
  title={Classification of Contradictory Opinions in Text Using Deep Learning Methods},
  author={Ogul, Iskender Ulgen},
  year={2020},
  school={Izmir Institute of Technology (Turkey)}
}
```

## Research Context

This repository presents a practical implementation of the research described in the references. It explores how different text representation methods perform on Natural Language Inference tasks.

The research project compares several approaches to representing text meaning:
- Traditional word embeddings (Word2Vec, GloVe, Spacy)
- spaCy's built-in word vectors
- BERT's context-aware embeddings


These different techniques are tested across NLI Benchmark datasets (SNLI, MultiNLI, and Adversarial NLI) to understand their strengths and weaknesses. The code shows how static word embeddings differ from contextual embeddings when determining if two sentences contradict each other or if one logically follows from the other. This work provides clear, reproducible examples that while making the methods accessible for further exploration and improvement.