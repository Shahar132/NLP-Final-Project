"""Text cleaning and spaCy-based tokenization, lemmatization, and POS tagging."""

import re

import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


def clean_text(text):
    """Strip HTML-like tags, punctuation, and newlines from raw text."""
    text = text.replace('\n', ' ')
    text = re.sub(r'<.*?>', ' ', text)  # removes HTML-like tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove punctuation
    return text


def preprocess_review(text):
    """Clean, tokenize, lemmatize, and POS-tag a single piece of text using spaCy."""
    if not hasattr(preprocess_review, "counter"):
        preprocess_review.counter = 0  # initialize counter

    text = clean_text(text.lower())
    doc = nlp(text)

    result = {
        'tokens': [token.text for token in doc],
        'lemmas': [token.lemma_ for token in doc],
        'pos_tags': [token.pos_ for token in doc]
    }

    # Print first 3 examples
    if preprocess_review.counter < 3:
        print(f"\n--- Example {preprocess_review.counter + 1} ---")
        print("Original text:", text)
        print("Tokens:", result['tokens'])
        print("Lemmas:", result['lemmas'])
        print("POS tags:", result['pos_tags'])
        print()
        preprocess_review.counter += 1

    return result
