# NLP Sentiment Analysis Project

Sentiment analysis on IMDb movie reviews, comparing a classic **Word2Vec + LSTM** pipeline against a **fine-tuned Transformer (DistilBERT)**, with an additional experiment on how training-data order (original / shuffled / reversed) affects each model's performance.

Developed as a final project for an NLP course.

---

## Overview

- **Word embeddings:** a custom Word2Vec model trained on IMDb reviews, a Wikipedia text corpus, or a combination of both.
- **Sequence model:** an LSTM classifier built on top of the Word2Vec embeddings.
- **Transformer model:** DistilBERT (`distilbert-base-uncased`), fine-tuned for binary sequence classification.
- **Order experiment:** both the LSTM and the Transformer are trained and evaluated three times each — on the data in its original order, shuffled, and reversed — to check whether sample order affects the resulting model.
- **Metrics:** Accuracy, F1-score, and confusion matrix, reported for every configuration.

---

## Pipeline

1. **Data loading**
   - IMDb reviews are loaded and a balanced sample is drawn (2,500 positive + 2,500 negative, by default).
   - The Wikipedia corpus is loaded as plain text lines (8,000 lines, by default).

2. **Preprocessing**
   - Text is lowercased, HTML-like tags and punctuation are stripped.
   - spaCy (`en_core_web_sm`) is used for tokenization, lemmatization, and POS tagging.

3. **Word2Vec training**
   - A Word2Vec model is trained on the lemmatized tokens.
   - The token source is configurable (`w2v_source`): IMDb only, Wikipedia only, or both combined (the default) — combining general Wikipedia text with domain-specific movie-review language.

4. **LSTM model**
   - A non-trainable embedding layer initialized from the Word2Vec vectors, followed by `SpatialDropout1D(0.2)`, an `LSTM(128)` layer (dropout 0.2, recurrent dropout 0.2), and a sigmoid output layer.
   - Trained with binary cross-entropy / Adam, 5 epochs, batch size 32, with an 80/20 train-test split (stratified) and a 20% validation split during training.

5. **Transformer model**
   - `distilbert-base-uncased` fine-tuned via the Hugging Face `Trainer` API for 2 epochs (batch size 16 train / 64 eval), with per-epoch evaluation.

6. **Order comparison**
   - Steps 4 and 5 are each repeated three times (original, shuffled, reversed data order) to compare the resulting accuracy, F1-score, and confusion matrix across orderings.

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

Main dependencies used by the script: `pandas`, `numpy`, `scikit-learn`, `spacy`, `gensim`, `tensorflow`, `transformers`, `datasets`, `tqdm`.

---

## Usage

Once the datasets are in place, run:

```bash
python main.py
```

The sample sizes (`positive_samples`, `negative_samples`, `rows_from_wiki`) and the Word2Vec source (`w2v_source`) can be adjusted inside `main()` for faster or more thorough runs.

The script will:

1. Load and inspect the IMDb dataset and the Wikipedia corpus.
2. Preprocess both (tokenization, lemmatization, POS tagging).
3. Train Word2Vec embeddings on the combined corpus.
4. Train and evaluate the LSTM model in all three data orders.
5. Fine-tune and evaluate the DistilBERT model in all three data orders.
6. Print accuracy, F1-score, and confusion matrices for every run.

---

## Notes

- Warnings and TensorFlow logging are suppressed by default, and progress bars are disabled, to keep console output focused on the printed metrics — this can be re-enabled by editing the flags at the top of the script.
- Both the LSTM and Transformer results are printed directly to the console rather than saved to a file; no trained weights or result logs are included in this repository.

---

## Authors

- Maayan Boni
- Shahar Eliyahu
