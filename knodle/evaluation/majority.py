from typing import Dict

import numpy as np
from sklearn.metrics import classification_report

from knodle.transformation.majority import probabilities_to_majority_vote, z_t_matrices_to_probs


def majority_sklearn_report(
        rule_matches_z: np.array, mapping_rules_labels_t: np.array, labels_y: np.array
) -> Dict:
    rule_counts_probs = z_t_matrices_to_probs(rule_matches_z, mapping_rules_labels_t)
    majority_y = np.apply_along_axis(
        probabilities_to_majority_vote, axis=1, arr=rule_counts_probs, ties_strategy="random"
    )
    sklearn_report = classification_report(labels_y, majority_y, output_dict=True)
    return sklearn_report


def sklearn_report_to_knodle_report(sklearn_report: Dict, prefix: str = None):
    if prefix is None:
        report = {
            f"accuracy": sklearn_report["accuracy"],
            f"macro_f1": sklearn_report["macro avg"]["f1-score"],
            f"weighted_f1": sklearn_report["weighted avg"]["f1-score"],
        }
    else:
        report = {
            f"{prefix}accuracy": sklearn_report["accuracy"],
            f"{prefix}macro_f1": sklearn_report["macro avg"]["f1-score"],
            f"{prefix}weighted_f1": sklearn_report["weighted avg"]["f1-score"],
        }
    return report
