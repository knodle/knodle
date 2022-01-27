import numpy as np

from typing import Dict, List
from sklearn.metrics import precision_score, recall_score, f1_score


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


def evaluate_multilabel(true_labels: Dict, predicted_probs: Dict, threshold: float, labels_indicator: List[str]):
    """
    Calculates precision, recall and F1 scores for multilabel classification results. The scores are macro
    averaged over the instances
    Args:
        true_labels: Dictionary with instance ids as keys. The values are lists with the string labels
        predicted_probs: Dictionary with instance ids as keys and predicted probabilities as values
        threshold: Number from 0 to 1 to be used as inference threshold
        labels_indicator: List with all the existing labels in the dataset, used to map each label to its id
        and corresponding predicted probability
    Returns: Average precision, recall and F1
    """

    assert true_labels.keys() == predicted_probs.keys()

    # Prepare ground truth
    y_true = encode_to_binary(list(true_labels.values()), labels_indicator)

    # Prepare predictions
    predicted_labels = get_predicted_labels(list(predicted_probs.values()), threshold, labels_indicator)
    y_pred = encode_to_binary(predicted_labels, labels_indicator)

    # Evaluate
    p = precision_score(y_true, y_pred, average="samples")
    r = recall_score(y_true, y_pred, average="samples")
    f1 = f1_score(y_true, y_pred, average="samples")

    return {"precision": p, "recall": r, "f1": f1}
