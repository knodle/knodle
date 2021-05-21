import numpy as np

from knodle.transformation.labels import label_ids_to_labels


def test_label_ids_to_labels():
    prediction_ids = np.array([0, 1, 1, 0])
    gold_label_ids = np.array([1, 0, 0, 1])

    labels2ids = {
        0: "first",
        1: "second"
    }

    predictions_truth = ["first", "second", "second", "first"]
    gold_labels_truth = ["second", "first", "first", "second"]

    predictions, gold_labels = label_ids_to_labels(prediction_ids, gold_label_ids, labels2ids)

    assert predictions == predictions_truth
    assert gold_labels == gold_labels_truth
