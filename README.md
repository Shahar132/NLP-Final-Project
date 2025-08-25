# NLP Sentiment Analysis Project

This project was developed as part of an NLP course.  
It implements **sentiment analysis** on IMDb movie reviews, with optional enrichment using a Wikipedia text corpus.  
The project compares performance between **LSTM with Word2Vec embeddings** and a **Transformer model (DistilBERT)**.

## Overview
- **Algorithms:**  
  - Word2Vec for word embeddings.  
  - LSTM for sequence modeling.  
  - DistilBERT for Transformer-based classification.  
- **Goal:**  
  Analyze sentiment (positive/negative) in IMDb reviews, and evaluate whether training order (original, shuffled, reversed) affects performance.  
- **Metrics Used:**  
  Accuracy, F1-score, and Confusion Matrix.

## Datasets
This project uses two datasets that are **not included** in the repository due to GitHub file size limitations:

1. [IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
   - 50,000 movie reviews labeled as *positive* or *negative*.  
   - Commonly used for binary sentiment classification.  

2. [Wikipedia Text Corpus (AllCombined)](https://www.kaggle.com/)  
   - Large English text dump (about 1M lines).  
   - In this project, a subset of **8,000 Wikipedia lines + 5,000 IMDb reviews** was used to train Word2Vec.  

### How to Prepare
1. Download both datasets from the Kaggle links above.  
2. Place them in the **project root folder**:  
   - `IMDB Dataset.csv`  
   - `AllCombined.txt`  
3. Make sure they are listed in `.gitignore` so they won’t be pushed to GitHub.  

## Installation
Clone the repository and install dependencies:

git clone https://github.com/your-username/nlp-sentiment-analysis.git  
cd nlp-sentiment-analysis  
pip install -r requirements.txt  

Make sure you also download **spaCy English model**:  
python -m spacy download en_core_web_sm  

## Usage
After placing the datasets in the project root, run:

python main.py  

The script will:  
1. Load IMDb dataset + Wikipedia corpus.  
2. Preprocess the data (tokenization, lemmatization, POS tagging).  
3. Train Word2Vec embeddings.  
4. Train and evaluate an LSTM model (original/shuffled/reversed).  
5. Train and evaluate a Transformer (DistilBERT).  
6. Print accuracy, F1, and confusion matrices for comparison.  

## Authors
- Maayan Boni  
- Shahar Eliyahu
