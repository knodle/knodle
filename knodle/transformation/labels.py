from typing import Dict
import numpy as np

def translate_predictions(predictions: np.ndarray, labels: np.ndarray, ids2labels: Dict) -> Dict:
    """
    Prepare data for string label based evaluation.
    :param predictions: predicted label ids
    :param labels: gold label ids
    :param ids2labels: translation dictionary from string labels to label ids
    :return:
    """
    predictions_idx = predictions.astype(int).tolist()
    labels_idx = labels.astype(int).tolist()

    predictions = [ids2labels[p] for p in predictions_idx]
    test_labels = [ids2labels[p] for p in labels_idx]
    return predictions, test_labels


# TODO: add function adding "other" class, if no label is given.