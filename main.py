"""
# Authors:
Maayan Boni
Shahar Eliyahu 
"""

# Final Project - NLP - Sentiment Analysis using IMDb Dataset + Wikipedia Corpus.
# Dataset 1: IMDb Movie Reviews Dataset.
# Source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# Description: This dataset contains 50,000 movie reviews labeled as either 'positive' or 'negative'.
# The dataset is balanced and commonly used for binary sentiment classification tasks in NLP.

# Dataset 2: Wikipedia Text Corpus (Allcombined).
# Source: A large Wikipedia dump from Kaggle containing nearly 1 million lines of general English text.
# Usage: While the full Wikipedia corpus was used for experimentation and exploration,
# the main Word2Vec training for the core LSTM sentiment analysis was conducted using a subset:
# 8,000 lines from the Wikipedia corpus and 5,000 labeled IMDb reviews (2,500 positive + 2,500 negative).
# This combined corpus aimed to enhance word embeddings by blending general and domain-specific language.


# For the main experiments, we used 5,000 IMDb reviews and 8,000 Wikipedia lines.
# Since processing this data can be time-consuming, you can adjust the sample sizes in main() to allow faster runs time.




# ️ Note:
#-----------
# If you want to see warnings and progress bars (e.g., from TensorFlow or tqdm),
# simply comment out the following lines by adding a '#' at the beginning of each line.
# This setup is currently used to keep the output clean and focused.
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#-----------


# Import necessary libraries
from tqdm import tqdm
tqdm.pandas(disable=True) # change to False to see progress bars ,true to disable them
import pandas as pd
import spacy
from gensim.models import Word2Vec
import random
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import numpy as np
import re




# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


#------------------------
# Functions for loading and inspecting data
#------------------------
def load_and_inspect_data(filepath,positive_samples=2500, negative_samples=2500):
    All_df = pd.read_csv(filepath)
    positive_df = All_df[All_df['sentiment'] == 'positive'].sample(n=positive_samples, random_state=42)
    negative_df = All_df[All_df['sentiment'] == 'negative'].sample(n=negative_samples, random_state=42)
    df = pd.concat([positive_df, negative_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(" First few rows of the imdb dataset:")
    print(df.head())
    print("\n Missing values in each column:")
    print(df.isnull().sum())
    print("\n Sentiment distribution:")
    print(df['sentiment'].value_counts())
    return df


#------------------------
# Functions for loading and preprocessing text data
#------------------------
def load_text_file_to_dataframe(filepath,number_of_rows=800):
    with open(filepath, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    df = pd.DataFrame(lines, columns=['text'])
    df.head(number_of_rows)
    print(f"\n Loaded {len(df)} rows from text (Wikipedia) file.")
    print(" First few rows of the text file:\n")
    print(df.head())

    return df

#clean text function
def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'<.*?>', ' ', text)  # removes HTML-like tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove punctuation
    return text


#------------------------
# B-D - Text Preprocessing
#------------------------

def preprocess_review(text):
    if not hasattr(preprocess_review, "counter"):
        preprocess_review.counter = 0  # initialize counter

    text = clean_text(text.lower())
    doc = nlp(text)

    result = {
        'tokens': [token.text for token in doc],
        'lemmas': [token.lemma_ for token in doc],
        'pos_tags': [token.pos_ for token in doc]
    }

    # Print first 5 examples
    if preprocess_review.counter < 3:
        print(f"\n--- Example {preprocess_review.counter + 1} ---")
        print("Original text:", text)
        print("Tokens:", result['tokens'])
        print("Lemmas:", result['lemmas'])
        print("POS tags:", result['pos_tags'])
        print()
        preprocess_review.counter += 1

    return result




#------------------------
# E - Train Word2Vec embeddings
#------------------------

def train_word2vec(token_lists, vector_size=100, window=5, min_count=2, workers=1, epochs=10):
    model = Word2Vec(vector_size=vector_size, window=window,
                     min_count=min_count, workers=workers)

    print("Building vocabulary")
    model.build_vocab(token_lists)
    print(f"Vocabulary size: {len(model.wv)}")

    print(f"Starting Word2Vec training for {epochs} epochs")
    model.train(token_lists, total_examples=len(token_lists), epochs=epochs)

    print("Word2Vec model trained successfully.")
    return model



#*******************
# LSTM
#*******************

#------------------------
# F - Build and Train LSTM model using Word2Vec embeddings
#------------------------
def train_lstm_model(df, w2v_model, max_len=100, order='original'):
    print(f"\nStarting LSTM model training ({order} order)")

    # Convert sentiment labels to binary
    y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values
    X = df['review'].values

    # Optional: shuffle or reverse
    if order == 'shuffled':
        indices = list(range(len(X)))
        random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    elif order == 'reversed':
        X = X[::-1]
        y = y[::-1]

    # Train-test split
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Tokenization
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_texts)
    X_train_seq = tokenizer.texts_to_sequences(X_train_texts)
    X_test_seq = tokenizer.texts_to_sequences(X_test_texts)

    # Padding
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    # Embedding matrix
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = w2v_model.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    missing_words = []
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
        else:
            missing_words.append(word)

    missing_count = len(missing_words)
    total_words = len(tokenizer.word_index)
    missing_ratio = (missing_count / total_words) * 100

    print(f"\nMissing words in Word2Vec: {missing_count} out of {total_words}")
    print(f"Missing word ratio: {missing_ratio:.2f}%")
    print("Sample missing words (first 10):")
    print(missing_words[:10])

    # Build LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_len,
                        trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("\nTraining LSTM model")
    model.fit(X_train_pad, y_train, validation_split=0.2, epochs=5, batch_size=32, verbose=1)

    # Evaluate
    y_pred_prob = model.predict(X_test_pad)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    print(f"\nLSTM ({order}) - Accuracy: {metrics['accuracy']:.4f}")
    print(f"LSTM ({order}) - F1 Score: {metrics['f1_score']:.4f}")
    print(f"LSTM ({order}) - Confusion Matrix:\n{metrics['confusion_matrix']}")

    return metrics, model


# ------------------------
# I - Compare LSTM models with different data orders
# ------------------------
def compare_data_orders(df, w2v_model):
    print("\n Comparing LSTM models with different data orders")

    # Train and evaluate with original order
    lstm_metrics_original, _ = train_lstm_model(df, w2v_model, order='original')


    # Train and evaluate with shuffled order
    lstm_metrics_shuffled, _ = train_lstm_model(df, w2v_model, order='shuffled')

    # Train and evaluate with reversed order
    lstm_metrics_reversed, _ = train_lstm_model(df, w2v_model, order='reversed')

    # Print comparison results
    print("\n === Summary of LSTM Results ===")
    print(
        f" Original Order - Accuracy: {lstm_metrics_original['accuracy']:.4f}, F1 Score: {lstm_metrics_original['f1_score']:.4f}")
    print(
        f" Shuffled Order - Accuracy: {lstm_metrics_shuffled['accuracy']:.4f}, F1 Score: {lstm_metrics_shuffled['f1_score']:.4f}")
    print(
        f" Reversed Order - Accuracy: {lstm_metrics_reversed['accuracy']:.4f}, F1 Score: {lstm_metrics_reversed['f1_score']:.4f}")



#*******************
# Transformer (DistilBERT)
#*******************
#------------------------
# G-H - Train and Evaluate Transformer (DistilBERT) model
#------------------------

# Compute evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'confusion_matrix': confusion_matrix(labels, preds)
    }


# Train Transformer model using HuggingFace Trainer and PyTorch backend
def train_transformer_model(df, order='original'):
    print(f"\n Starting Transformer (DistilBERT) training ({order} order)")

    # Change data order if specified
    if order == 'shuffled':
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    elif order == 'reversed':
        df = df.iloc[::-1].reset_index(drop=True)

    # Map sentiment labels to binary values
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    texts = df['review'].tolist()
    labels = df['label'].tolist()

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42)

    # Load tokenizer and tokenize text
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    test_encodings = tokenizer(X_test, truncation=True, padding=True)

    # Create HuggingFace datasets
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'label': y_train
    })
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'label': y_test
    })

    # Load pre-trained DistilBERT model for binary classification
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        save_strategy="no",
        load_best_model_at_end=False
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    print(f"\n Transformer ({order} order) - Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f" Transformer ({order} order) - F1 Score: {eval_results['eval_f1']:.4f}")
    print(f" Transformer ({order} order) - Confusion Matrix:\n {eval_results['eval_confusion_matrix']}")

    return eval_results




#------------------------
# J - Compare Transformer models with different data orders
#------------------------
def compare_transformer_orders(df):
    print("\n Comparing Transformer models with different data orders")

    results = {}

    # Train and evaluate with different data orders
    for order in ['original', 'shuffled', 'reversed']:
        results[order] = train_transformer_model(df, order=order)

    # Print summary of results
    print("\n === Summary of Transformer Results ===")
    for order in ['original', 'shuffled', 'reversed']:
        acc = results[order]['eval_accuracy']
        f1 = results[order]['eval_f1']
        cm = results[order]['eval_confusion_matrix']
        print(f"\n Order: {order.capitalize()}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")




#------------------------
# Main execution function
#------------------------


def main():
    print("\nStarting the NLP Sentiment Analysis Project\n")
    #------
    # Choose Word2Vec source: "imdb", "wiki", or "both" to train the Word2Vec model.
    #-----
    w2v_source = "imdb+wiki"  #  change to: "imdb", "wiki", or "imdb+wiki"

    # Load IMDb dataset
    positive_samples = 2500  # Number of positive samples to use from IMDb dataset.#change to adjust IMDB positive samples
    negative_samples = 2500  # Number of negative samples to use from IMDb dataset.change to adjust IMDB negative samples


    df = load_and_inspect_data("IMDB Dataset.csv",negative_samples=negative_samples, positive_samples=positive_samples)
    df_for_w2v = df.copy().rename(columns={"review": "text"})
    print(f"\nLoaded {len(df)} rows from IMDb dataset.")


    rows_from_wiki = 8000 # Number of rows to use from Wikipedia corpus for Word2Vec training.Change for different runtimes for testing

    # Load external Wikipedia corpus
    df_corpus = load_text_file_to_dataframe("AllCombined.txt",).head(rows_from_wiki)  # Limit rows for faster processing.
    print(f"\nUse {rows_from_wiki} rows from Wikipedia corpus.")

    print("Example for preprocessing text (tokenization, lemmatization, and POS tagging):")
    # Preprocess IMDb and Wikipedia datasets separately.
    df_for_w2v['processed'] = df_for_w2v['text'].progress_apply(preprocess_review)
    df_corpus['processed'] = df_corpus['text'].progress_apply(preprocess_review)

    #--------
    # Choose data for Word2Vec training.
    #--------
    print(f"\n Preparing token lists for Word2Vec from {w2v_source} source\n")
    if w2v_source == "imdb":
        token_lists = df_for_w2v['processed'].apply(lambda x: x['lemmas']).tolist()
    elif w2v_source == "wiki":
        token_lists = df_corpus['processed'].apply(lambda x: x['lemmas']).tolist()
    else:  # both
        token_lists = (
            df_for_w2v['processed'].apply(lambda x: x['lemmas']).tolist() +
            df_corpus['processed'].apply(lambda x: x['lemmas']).tolist()
        )

    # Train Word2Vec model
    w2v_model = train_word2vec(token_lists)

    # Prepare IMDb reviews for LSTM input (cleaned lemmas)
    df['processed'] = df_for_w2v['processed']
    df['review'] = df['processed'].apply(lambda x: ' '.join(x['lemmas']))

    # Train and evaluate LSTM
    compare_data_orders(df, w2v_model)

    # Optional: Train and evaluate Transformer
    compare_transformer_orders(df)



# Run the main function
if __name__ == "__main__":
    main()