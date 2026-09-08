"""Shared evaluation helpers for computing classification metrics (Accuracy, F1, Confusion Matrix)."""

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_classification_metrics(y_true, y_pred):
    """Compute accuracy, F1 score, and confusion matrix for binary classification.

    Used directly by the LSTM evaluation, and wrapped by compute_metrics() below
    for the HuggingFace Trainer.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def compute_metrics(pred):
    """Compute evaluation metrics for a HuggingFace Trainer prediction output.

    HuggingFace's Trainer expects a metrics function with this exact signature
    (a single `pred` object with .label_ids and .predictions), and expects the
    F1 key to be named 'f1' rather than 'f1_score'.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    metrics = compute_classification_metrics(labels, preds)

    return {
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1_score'],
        'confusion_matrix': metrics['confusion_matrix']
    }
