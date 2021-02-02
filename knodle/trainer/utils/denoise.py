import numpy as np

from random import randint


def get_majority_vote_probs_with_no_rel(
        rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray, no_match_class: int = None
) -> np.ndarray:
    """
    This function calculates a majority vote probability for all rule_matches_z. The difference from simple
    get_majority_vote_probs function is the following: samples, where no rules matched (that is, all elements in
     the corresponding raw in rule_matches_z matrix equal 0), are assigned to no_match_class (that is, a value in the
     corresponding column in rule_counts_probs matrix is changed to 1)
    Args:
        rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
        mapping_rules_labels_t: Mapping of rules to labels, binary encoded. Shape: rules x classes
        no_match_class:
    Returns: Array with majority vote decision. Shape: instances x classes
    """
    if rule_matches_z.shape[1] != mapping_rules_labels_t.shape[0]:
        raise ValueError("Dimensions mismatch!")

    rule_counts = np.matmul(rule_matches_z, mapping_rules_labels_t)
    if no_match_class is not None:
        rule_counts[~rule_counts.any(axis=1), no_match_class] = 1
    rule_counts_probs = rule_counts / rule_counts.sum(axis=1).reshape(-1, 1)

    rule_counts_probs[np.isnan(rule_counts_probs)] = 0
    return rule_counts_probs


def activate_neighbors(
        rule_matches_z: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """
    Find all closest neighbors and take the same label ids
    Args:
        rule_matches_z: All rule matches. Shape: instances x rules
        indices: Neighbor indices from knn
    Returns:
    """
    new_z_matrix = np.zeros(rule_matches_z.shape)
    for index, sample in enumerate(rule_matches_z):
        neighbors = indices[index].astype(int)
        neighborhood_z = rule_matches_z[neighbors, :]

        activated_lfs = neighborhood_z.sum(axis=0)  # Add all lf activations

        # All with != 0 are valid. We could also do some division here for weighted
        activated_lfs = np.where(activated_lfs > 0)[0]
        new_z_matrix[index, activated_lfs] = 1

    return new_z_matrix
