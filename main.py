"""
# Authors:
Maayan Boni
Shahar Eliyahu
"""

import os
import warnings

# Keep console output clean
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
from tqdm import tqdm

from data_loader import load_and_inspect_data, load_text_file_to_dataframe
from embeddings import train_word2vec
from lstm_model import compare_data_orders
from preprocessing import preprocess_review
from transformer_model import compare_transformer_orders


tqdm.pandas(disable=True)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# True  -> run with a small built-in dataset for testing
# False -> run the original project using the real datasets
TEST_MODE = True

# Running DistilBERT can take much longer and may download the model.
# Keep False for the first smoke test.
RUN_TRANSFORMER = True

W2V_SOURCE = "imdb+wiki"

IMDB_PATH = "IMDB Dataset.csv"
WIKIPEDIA_PATH = "AllCombined.txt"

POSITIVE_SAMPLES = 2500
NEGATIVE_SAMPLES = 2500
WIKI_ROWS = 8000


def create_test_dataset():
    """
    Create a very small balanced sentiment dataset.

    This dataset is only used to verify that the refactored project
    runs correctly without requiring the original datasets.
    """

    positive_reviews = [
        "I really enjoyed this movie and the acting was excellent.",
        "This film was amazing and I would happily watch it again.",
        "A wonderful story with great characters and beautiful scenes.",
        "The movie was entertaining from beginning to end.",
        "Excellent performance and a very enjoyable story.",
        "I loved this movie and thought the plot was fantastic.",
        "A great film with strong acting and an interesting story.",
        "This was one of the best movies I have watched recently.",
        "Very enjoyable movie with excellent characters.",
        "The story was touching and the acting was very good.",
        "I had a great time watching this film.",
        "The movie was exciting and beautifully made.",
        "Fantastic movie with a strong story.",
        "I really liked the characters and the ending.",
        "An excellent and entertaining film.",
        "The acting was great and the story kept me interested.",
        "A very good movie that I would recommend.",
        "I enjoyed every part of this film.",
        "The movie had a wonderful story and strong performances.",
        "A fun and enjoyable movie experience.",
    ]

    negative_reviews = [
        "I really disliked this movie and the acting was terrible.",
        "This film was boring and I would not watch it again.",
        "A disappointing story with weak characters.",
        "The movie was boring from beginning to end.",
        "Poor performance and a very dull story.",
        "I hated this movie and thought the plot was terrible.",
        "A bad film with weak acting and an uninteresting story.",
        "This was one of the worst movies I have watched recently.",
        "Very disappointing movie with terrible characters.",
        "The story was boring and the acting was very bad.",
        "I did not enjoy watching this film.",
        "The movie was slow and poorly made.",
        "Terrible movie with a weak story.",
        "I really disliked the characters and the ending.",
        "A poor and disappointing film.",
        "The acting was bad and the story was not interesting.",
        "A very bad movie that I would not recommend.",
        "I disliked almost every part of this film.",
        "The movie had a weak story and poor performances.",
        "A boring and disappointing movie experience.",
    ]

    df = pd.DataFrame(
        {
            "review": positive_reviews + negative_reviews,
            "sentiment": (
                ["positive"] * len(positive_reviews)
                + ["negative"] * len(negative_reviews)
            ),
        }
    )

    # Shuffle while keeping the run reproducible
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    print("\nStarting the NLP Sentiment Analysis Project\n")

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    if TEST_MODE:
        print("TEST MODE enabled")
        print("Using a small built-in dataset instead of external files.\n")

        df = create_test_dataset()

        # No Wikipedia corpus is required in test mode
        df_corpus = None
        w2v_source = "imdb"

    else:
        print("FULL PROJECT MODE\n")

        df = load_and_inspect_data(
            IMDB_PATH,
            positive_samples=POSITIVE_SAMPLES,
            negative_samples=NEGATIVE_SAMPLES,
        )

        w2v_source = W2V_SOURCE
        df_corpus = None

        if w2v_source in ("wiki", "imdb+wiki"):
            df_corpus = load_text_file_to_dataframe(
                WIKIPEDIA_PATH
            ).head(WIKI_ROWS)

            print(f"\nUsing {len(df_corpus)} Wikipedia rows.")

    print(f"Loaded {len(df)} IMDb reviews.")

    # --------------------------------------------------
    # Text preprocessing
    # --------------------------------------------------

    print(
        "\nPreprocessing text "
        "(tokenization, lemmatization and POS tagging)..."
    )

    df_for_w2v = df.copy().rename(columns={"review": "text"})

    df_for_w2v["processed"] = (
        df_for_w2v["text"]
        .progress_apply(preprocess_review)
    )

    if df_corpus is not None:
        df_corpus["processed"] = (
            df_corpus["text"]
            .progress_apply(preprocess_review)
        )

    # --------------------------------------------------
    # Prepare Word2Vec training corpus
    # --------------------------------------------------

    print(
        f"\nPreparing token lists for Word2Vec "
        f"from {w2v_source} source\n"
    )

    imdb_tokens = (
        df_for_w2v["processed"]
        .apply(lambda x: x["lemmas"])
        .tolist()
    )

    if w2v_source == "imdb":
        token_lists = imdb_tokens

    elif w2v_source == "wiki":
        if df_corpus is None:
            raise ValueError(
                "Wikipedia dataset is required when W2V_SOURCE='wiki'."
            )

        token_lists = (
            df_corpus["processed"]
            .apply(lambda x: x["lemmas"])
            .tolist()
        )

    elif w2v_source == "imdb+wiki":
        if df_corpus is None:
            raise ValueError(
                "Wikipedia dataset is required "
                "when W2V_SOURCE='imdb+wiki'."
            )

        wiki_tokens = (
            df_corpus["processed"]
            .apply(lambda x: x["lemmas"])
            .tolist()
        )

        token_lists = imdb_tokens + wiki_tokens

    else:
        raise ValueError(
            "W2V_SOURCE must be 'imdb', 'wiki', or 'imdb+wiki'."
        )

    # --------------------------------------------------
    # Word2Vec
    # --------------------------------------------------

    w2v_model = train_word2vec(token_lists)

    # --------------------------------------------------
    # Prepare reviews for LSTM
    # --------------------------------------------------

    df["processed"] = df_for_w2v["processed"]

    df["review"] = df["processed"].apply(
        lambda x: " ".join(x["lemmas"])
    )

    # --------------------------------------------------
    # LSTM experiments
    # --------------------------------------------------

    compare_data_orders(df, w2v_model)

    # --------------------------------------------------
    # Transformer experiments
    # --------------------------------------------------

    if RUN_TRANSFORMER:
        compare_transformer_orders(df)
    else:
        print(
            "\nTransformer experiment skipped "
            "(RUN_TRANSFORMER = False)."
        )

    print("\nProject execution completed successfully.")


if __name__ == "__main__":
    main()