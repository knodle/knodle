from typing import Dict

import numpy as np
from sklearn.metrics.classification import classification_report

from knodle.transformation.majority import probabilities_to_majority_vote, z_t_matrices_to_majority_vote_probs


def majority_sklearn_report(
        rule_matches_z: np.array, mapping_rules_labels_t: np.array, labels_y: np.array
) -> Dict:
    rule_counts_probs = z_t_matrices_to_majority_vote_probs(rule_matches_z, mapping_rules_labels_t)

    kwargs = {"choose_random_label": True}
    majority_y = np.apply_along_axis(probabilities_to_majority_vote, axis=1, arr=rule_counts_probs, **kwargs)

    sklearn_report = classification_report(labels_y, majority_y, output_dict=True)

    return sklearn_report
