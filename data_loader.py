"""Functions for loading and inspecting the IMDb and Wikipedia datasets."""

import pandas as pd


def load_and_inspect_data(filepath, positive_samples=2500, negative_samples=2500):
    """Load the IMDb dataset, sample a balanced subset, and print a quick inspection summary."""
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


def load_text_file_to_dataframe(filepath, number_of_rows=800):
    """Load a plain-text corpus (e.g. Wikipedia lines) into a single-column DataFrame."""
    with open(filepath, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    df = pd.DataFrame(lines, columns=['text'])
    df.head(number_of_rows)
    print(f"\n Loaded {len(df)} rows from text (Wikipedia) file.")
    print(" First few rows of the text file:\n")
    print(df.head())

    return df
