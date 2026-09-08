"""Word2Vec training on preprocessed token lists."""

from gensim.models import Word2Vec


def train_word2vec(token_lists, vector_size=100, window=5, min_count=2, workers=1, epochs=10):
    """Train a Word2Vec model on the given list of tokenized/lemmatized documents."""
    model = Word2Vec(vector_size=vector_size, window=window,
                      min_count=min_count, workers=workers)

    print("Building vocabulary")
    model.build_vocab(token_lists)
    print(f"Vocabulary size: {len(model.wv)}")

    print(f"Starting Word2Vec training for {epochs} epochs")
    model.train(token_lists, total_examples=len(token_lists), epochs=epochs)

    print("Word2Vec model trained successfully.")
    return model
