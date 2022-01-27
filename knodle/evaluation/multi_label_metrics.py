import numpy as np

from typing import Dict, List
from sklearn.metrics import precision_recall_fscore_support


def encode_to_binary(labels: List[List[str]], labels_indicator: List[str]):
    """
    Encodes the labels of each instance to binary vectors
    Args:
        labels: List with the labels for each instance (strings)
        labels_indicator: List with all the existing labels in the dataset, used to map each label to its id to
        create the binary vector
    Returns: Binary vectors for each instance
    """
    encoded_labels = []
    for label in labels:
        y = np.zeros(len(labels_indicator), dtype=float)

        for i, label_id in enumerate(labels_indicator):
            # Can leave zero if doesn't exist in ground truth, else make one
            if label_id in label:
                y[i] = 1

        encoded_labels.append(y)

    return np.array(encoded_labels)


def get_predicted_labels(probs: List[float], threshold: float, labels_indicator: List[str]):
    """
    Assigns to each instance the labels with a probability above/equal of the threshold
    Args:
        probs: List with the predicted probability for each instance
        threshold: Number from 0 to 1 to be used as inference threshold
        labels_indicator: List with all the existing labels in the dataset, used to map each label to its probability
    Returns: List with the assigned labels (strings)
    """
    predicted_labels = []

    for p in probs:
        indices = np.argwhere(np.array(p) >= threshold).flatten()
        predicted_labels.append([labels_indicator[index] for index in indices])

    return predicted_labels


def evaluate_multilabel(y_true: Dict, y_pred: Dict, threshold: float, ids2labels: List[str]):
    """
    Calculates precision, recall and F1 scores for multilabel classification results. The scores are macro
    averaged over the instances
    Args:
        y_true: Dictionary with instance ids as keys. The values are lists with the string labels
        y_pred: Dictionary with instance ids as keys and predicted probabilities as values
        threshold: Number from 0 to 1 to be used as inference threshold
        ids2labels: List with all the existing labels in the dataset, used to map each label to its id
        and corresponding predicted probability
    Returns: Average precision, recall and F1
    """

    assert y_true.keys() == y_pred.keys()

    # Prepare ground truth
    y_true = encode_to_binary(list(y_true.values()), ids2labels)

    # Prepare predictions
    predicted_labels = get_predicted_labels(list(y_pred.values()), threshold, ids2labels)
    y_pred = encode_to_binary(predicted_labels, ids2labels)

    # Evaluate
    return precision_recall_fscore_support(y_true, y_pred, average="samples")
