import numpy as np
from sklearn.metrics.classification import classification_report

from knodle.trainer.utils.denoise import get_majority_vote_labels
from knodle.evaluation.sklearn_utils import sklearn_report_to_knodle_report


def majority_report(rule_matches_z: np.array, mapping_rules_labels_t: np.array, labels_y: np.array, prefix: str = None):
    y_majority = get_majority_vote_labels(rule_matches_z, mapping_rules_labels_t)
    sklearn_report = classification_report(labels_y, y_majority, output_dict=True)

    return sklearn_report_to_knodle_report(sklearn_report, prefix)
