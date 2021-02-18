def translate_predictions(predictions: np.ndarray, labels: np.ndarray, labels2ids: Dict) -> Dict:
    """
    Prepare data for string label based evaluation.
    :param predictions: predicted label ids
    :param labels: gold label ids
    :param labels2ids: translation dictionary from string labels to label ids
    :return:
    """
    predictions_idx = predictions.astype(int).tolist()
    labels_idx = labels.astype(int).tolist()
    idx2labels = dict([(value, key) for key, value in labels2ids.items()])

    predictions = [idx2labels[p] for p in predictions_idx]
    test_labels = [idx2labels[p] for p in labels_idx]
    return predictions, test_labels


# TODO: add function adding "other" class, if no label is given.