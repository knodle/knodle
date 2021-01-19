import numpy as np


def get_majority_vote_probs(
    rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray
) -> np.ndarray:
    """
    This function calculates a majority vote probability for all rule_matches_z. First rule counts will be
    calculated,
    then a probability will be calculated by dividing the values row-wise with the sum. To counteract zero
    division
    all nan values are set to zero.
    Args:
        rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
        mapping_rules_labels_t: Mapping of rules to labels, binary encoded. Shape: rules x classes
    Returns: Array with majority vote decision. Shape: instances x classes

    """
    if rule_matches_z.shape[1] != mapping_rules_labels_t.shape[0]:
        raise ValueError("Dimensions mismatch!")

    rule_counts = np.matmul(rule_matches_z, mapping_rules_labels_t)
    rule_counts_probs = rule_counts / rule_counts.sum(axis=1).reshape(-1, 1)

    rule_counts_probs[np.isnan(rule_counts_probs)] = 0
    return rule_counts_probs
