from typing import Tuple

import numpy as np
from scipy import sparse as ss


def z_t_matrix_to_snorkel_matrix(rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray) -> np.ndarray:
    snorkel_matrix = -1 * np.ones(rule_matches_z.shape)

    if isinstance(rule_matches_z, ss.csr_matrix):
        rule_matches_z = rule_matches_z.toarray()

    z_to_t = np.argmax(mapping_rules_labels_t, axis=-1)
    print(z_to_t, type(z_to_t))
    if isinstance(mapping_rules_labels_t, ss.csr_matrix):
        # transform np.matrix to np.array
        z_to_t = np.array(z_to_t).flatten()

    for i in range(rule_matches_z.shape[0]):
        non_zero_idx = np.nonzero(rule_matches_z[i])[0]
        snorkel_matrix[i, non_zero_idx] = z_to_t[non_zero_idx]

    return snorkel_matrix


def prepare_empty_rule_matches(rule_matches_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove empty rows in rule matches for LabelModel.
    Args:
        rule_matches_z: input rule matches
    Returns:
        boolean mask indicating non-empty rows and filtered rule matches
    """
    # find empty rows of the rule matches
    non_zero_mask = rule_matches_z.sum(axis=1) != 0
    if isinstance(rule_matches_z, ss.csr_matrix):
        # transform np.matrix to np.array
        non_zero_mask = np.array(non_zero_mask).flatten()
    non_zero_indices = np.where(non_zero_mask)[0]

    # exclude empty rows from LabelModel input
    rule_matches_z = rule_matches_z[non_zero_indices]
    return non_zero_mask, rule_matches_z


def add_labels_for_empty_examples(
        label_probs_gen: np.ndarray, non_zero_mask: np.ndarray, output_classes: int, other_class_id: int
) -> np.ndarray:
    """
    Args:
        label_probs_gen: labels generated for the non-empty rows
        non_zero_mask:  boolean mask indicating which examples get the generated labels
        output_classes: number of output classes, with respect to the other_class
        other_class_id: id of the class for empty rows
    Returns: distribution of labels for both empty and non-empty rows (#instances x #classes)
    """
    # make dummy label distibutions for all of the examples
    # number of output classes is eventually t.shape[1]+1, if other class id should be added
    label_probs = np.zeros((non_zero_mask.shape[0], output_classes))

    # fill labels of the generative model
    label_probs[non_zero_mask, :label_probs_gen.shape[1]] = label_probs_gen

    # assign full probability to other class for empty rows
    label_probs[~non_zero_mask, other_class_id] = 1
    return label_probs
