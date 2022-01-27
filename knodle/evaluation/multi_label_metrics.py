import numpy as np

from typing import Dict, List
from sklearn.metrics import precision_recall_fscore_support


def encode_to_binary(labels: List[List[int]], num_labels: int) -> np.array:
    """
    Encodes the labels of each instance to binary vectors
    Args:
        labels: List with the labels for each instance (ids)
        num_labels: number of classes
    Returns: Binary vectors for each instance
    """
    encoded_labels = []
    for labeled_instance in labels:
        y = np.zeros(num_labels, dtype=float)

        for label in labeled_instance:
            # Can leave zero if doesn't exist in ground truth, else make one
            y[label] = 1

        encoded_labels.append(y)

    return np.array(encoded_labels)


def get_predicted_labels(probs: np.array, threshold: float) -> List:
    """
    Assigns to each instance the labels with a probability above/equal of the threshold
    Args:
        probs: List with the predicted probabilities for each instance
        threshold: Number from 0 to 1 to be used as inference threshold
    Returns: List with the assigned labels (ids)
    """
    predicted_labels = []

    for p in probs:
        indices = np.argwhere(p >= threshold).flatten()
        predicted_labels.append(indices)

    return predicted_labels


def evaluate_multi_label(
        y_true: List[List[int]], y_pred: np.ndarray, threshold: float, ids2labels: Dict
) -> Dict[str: float]:
    """
    Calculates precision, recall and F1 scores for multi-label classification results. The scores are macro
    averaged over the instances.
    Args:
        y_true: List that contains a list with the ids of ground truth labels for each instance
        y_pred: Numpy array with the predicted probabilities
        threshold: Number from 0 to 1 to be used as inference threshold
        ids2labels: Dictionary with ids and the corresponding labels (ids start from 0)
    Returns: Average precision, recall and F1
    """

    # Prepare ground truth
    y_true_binary = encode_to_binary(y_true, len(ids2labels))

    # Prepare predictions
    predicted_labels = get_predicted_labels(y_pred, threshold)
    y_pred_binary = encode_to_binary(predicted_labels, len(ids2labels))

    precision, recall, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average="samples")
    # Evaluate
    return {"precision": precision, "recall": recall, "f1": f1}
