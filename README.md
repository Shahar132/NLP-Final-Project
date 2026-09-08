# NLP Sentiment Analysis Project

Sentiment analysis on IMDb movie reviews, comparing a classic **Word2Vec + LSTM** pipeline against a **fine-tuned Transformer (DistilBERT)**, with an additional experiment on how training-data order (original / shuffled / reversed) affects each model's performance.

Developed as a final project for an NLP course. Refactored into a multi-file structure (see [Project Structure](#project-structure)) while keeping the exact same algorithms, hyperparameters, and outputs as the original single-file version.

---

## Overview

- **Word embeddings:** a custom Word2Vec model trained on IMDb reviews, a Wikipedia text corpus, or a combination of both.
- **Sequence model:** an LSTM classifier built on top of the Word2Vec embeddings.
- **Transformer model:** DistilBERT (`distilbert-base-uncased`), fine-tuned for binary sequence classification.
- **Order experiment:** both the LSTM and the Transformer are trained and evaluated three times each — on the data in its original order, shuffled, and reversed — to check whether sample order affects the resulting model.
- **Metrics:** Accuracy, F1-score, and confusion matrix, reported for every configuration.

---

## Project Structure

```text
NLP-Final-Project/
│
├── main.py                # Configuration + pipeline orchestration
├── data_loader.py          # Load IMDb CSV and Wikipedia text corpus
├── preprocessing.py        # Text cleaning + spaCy tokenization/lemmatization/POS tagging
├── embeddings.py            # Word2Vec training
├── lstm_model.py            # Word2Vec + LSTM model, order-comparison experiment
├── transformer_model.py     # DistilBERT model, order-comparison experiment
├── evaluation.py            # Shared accuracy/F1/confusion-matrix helpers
└── README.md
```

### How the modules interact

`main.py` sets configuration values (sample sizes, Word2Vec source) and calls into the other modules in sequence — it contains no modeling logic itself:

1. `data_loader` loads the IMDb reviews and the Wikipedia corpus.
2. `preprocessing` cleans and tokenizes both.
3. `embeddings` trains Word2Vec on the resulting tokens.
4. `lstm_model` builds and evaluates the LSTM classifier (calling into `evaluation` for metrics) across all three data orders.
5. `transformer_model` fine-tunes and evaluates DistilBERT (also calling into `evaluation`) across all three data orders.

`evaluation.py` has no dependencies on any other project module, so both model modules can depend on it without any circular imports.

---

## Datasets

The datasets are **not included** in this repository (file size / distribution limits):

1. **[IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)**
   50,000 movie reviews labeled `positive` / `negative` — a standard balanced dataset for binary sentiment classification.

2. **Wikipedia Text Corpus (`AllCombined.txt`)**
   A large Wikipedia text dump (roughly 1M lines of general English text), used to enrich the Word2Vec embeddings with broader language context beyond movie reviews.

### Setup

1. Download the IMDb dataset from the Kaggle link above, and obtain a Wikipedia text corpus (e.g. a Wikipedia dump/export) as a plain-text file.
2. Place both files in the project root:
   - `IMDB Dataset.csv`
   - `AllCombined.txt`
3. Keep them listed in `.gitignore` so they aren't pushed to GitHub.

---

## Installation

```bash
git clone https://github.com/your-username/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Main dependencies used by the project: `pandas`, `numpy`, `scikit-learn`, `spacy`, `gensim`, `tensorflow`, `transformers`, `datasets`, `tqdm`.

---

## Usage

Once the datasets are in place, run:

```bash
python main.py
```

The sample sizes (`positive_samples`, `negative_samples`, `rows_from_wiki`) and the Word2Vec source (`w2v_source`) can be adjusted directly inside `main()` for faster or more thorough runs.

The script will:

1. Load and inspect the IMDb dataset and the Wikipedia corpus.
2. Preprocess both (tokenization, lemmatization, POS tagging).
3. Train Word2Vec embeddings on the combined corpus.
4. Train and evaluate the LSTM model in all three data orders.
5. Fine-tune and evaluate the DistilBERT model in all three data orders.
6. Print accuracy, F1-score, and confusion matrices for every run.

---

## Notes

- Warnings and TensorFlow logging are suppressed by default, and progress bars are disabled, to keep console output focused on the printed metrics — this can be re-enabled by editing the flags at the top of `main.py`.
- Both the LSTM and Transformer results are printed directly to the console rather than saved to a file; no trained weights or result logs are included in this repository.

---

## Authors

- Maayan Boni
- Shahar Eliyahu
