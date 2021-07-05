from typing import Dict
import numpy as np


def label_ids_to_labels(predictions: np.ndarray, labels: np.ndarray, ids2labels: Dict) -> [np.ndarray, np.ndarray]:
    """
    Prepare data for string label based evaluation.

    Args:
        predictions: predicted label ids
        labels: gold label ids
        ids2labels: translation dictionary from string labels to label ids
    Returns:
        Two lists of class labels for predictions and gold labels
    """
    predictions_idx = predictions.astype(int).tolist()
    labels_idx = labels.astype(int).tolist()

    predictions = [ids2labels[p] for p in predictions_idx]
    gold_labels = [ids2labels[p] for p in labels_idx]
    return predictions, gold_labels
