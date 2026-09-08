"""LSTM sentiment classifier built on top of Word2Vec embeddings, and the
original/shuffled/reversed data-order comparison experiment.
"""

import random

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from evaluation import compute_classification_metrics


def _reorder_data(X, y, order):
    """Reorder the input arrays according to the requested order (original/shuffled/reversed)."""
    if order == 'shuffled':
        indices = list(range(len(X)))
        random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    elif order == 'reversed':
        X = X[::-1]
        y = y[::-1]
    return X, y


def _build_embedding_matrix(tokenizer, w2v_model):
    """Build an embedding matrix for the tokenizer's vocabulary from a trained Word2Vec model."""
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

    return embedding_matrix, vocab_size, embedding_dim


def _build_lstm_model(vocab_size, embedding_dim, embedding_matrix, max_len):
    """Construct and compile the LSTM model with a frozen Word2Vec embedding layer."""
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
    return model


def train_lstm_model(df, w2v_model, max_len=100, order='original'):
    """Train and evaluate the LSTM sentiment classifier for a given data order."""
    print(f"\nStarting LSTM model training ({order} order)")

    # Convert sentiment labels to binary
    # Convert sentiment labels to binary NumPy array
    y = (
        df["sentiment"]
        .map({"positive": 1, "negative": 0})
        .to_numpy(dtype=np.int64)
    )

    # Convert reviews to a standard NumPy object array
    X = (
        df["review"]
        .astype(str)
        .to_numpy(dtype=object)
    )

    # Optional: shuffle or reverse
    X, y = _reorder_data(X, y, order)

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
    embedding_matrix, vocab_size, embedding_dim = _build_embedding_matrix(tokenizer, w2v_model)

    # Build LSTM model
    model = _build_lstm_model(vocab_size, embedding_dim, embedding_matrix, max_len)

    print("\nTraining LSTM model")
    model.fit(X_train_pad, y_train, validation_split=0.2, epochs=5, batch_size=32, verbose=1)

    # Evaluate
    y_pred_prob = model.predict(X_test_pad)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    metrics = compute_classification_metrics(y_test, y_pred)

    print(f"\nLSTM ({order}) - Accuracy: {metrics['accuracy']:.4f}")
    print(f"LSTM ({order}) - F1 Score: {metrics['f1_score']:.4f}")
    print(f"LSTM ({order}) - Confusion Matrix:\n{metrics['confusion_matrix']}")

    return metrics, model


def compare_data_orders(df, w2v_model):
    """Train and evaluate the LSTM model with original, shuffled, and reversed data orders."""
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
