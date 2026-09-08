"""DistilBERT sentiment classifier fine-tuning, and the original/shuffled/reversed
data-order comparison experiment.
"""

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (DistilBertForSequenceClassification, DistilBertTokenizerFast,
                           Trainer, TrainingArguments)

from evaluation import compute_metrics


def _reorder_dataframe(df, order):
    """Reorder the DataFrame rows according to the requested order (original/shuffled/reversed)."""
    if order == 'shuffled':
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    elif order == 'reversed':
        df = df.iloc[::-1].reset_index(drop=True)
    return df


def _build_hf_datasets(X_train, y_train, X_test, y_test, tokenizer):
    """Tokenize the train/test texts and wrap them into HuggingFace Datasets."""
    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    test_encodings = tokenizer(X_test, truncation=True, padding=True)

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
    return train_dataset, test_dataset


def train_transformer_model(df, order='original'):
    """Fine-tune and evaluate the DistilBERT sentiment classifier for a given data order."""
    print(f"\n Starting Transformer (DistilBERT) training ({order} order)")

    # Change data order if specified
    df = _reorder_dataframe(df, order)

    # Map sentiment labels to binary values
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    texts = df['review'].tolist()
    labels = df['label'].tolist()

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42)

    # Load tokenizer and build HuggingFace datasets
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_dataset, test_dataset = _build_hf_datasets(X_train, y_train, X_test, y_test, tokenizer)

    # Load pre-trained DistilBERT model for binary classification
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
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


def compare_transformer_orders(df):
    """Fine-tune and evaluate the Transformer model with original, shuffled, and reversed data orders."""
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
