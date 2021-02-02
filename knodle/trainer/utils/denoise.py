import numpy as np

from random import randint


def sample_majority(probs: np.array, random_label: bool = True, default_label: int = -1):
    row_max = np.max(probs)
    num_occurrences = (row_max == probs).sum()
    if num_occurrences == 1:
        return np.argmax(probs)
    else:
        if random_label:
            return randint(0, probs.shape[0] - 1)
        return default_label


def get_majority_vote_labels(
        rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray, no_rule_label: int = -1
) -> np.array:
    """Computes the majority labels. If no clear "winner" is found, no_rule_label is used instead.
    Args:
        rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
        mapping_rules_labels_t: Mapping of rules to labels, binary encoded. Shape: rules x classes
        no_rule_label: Dummy label
    Returns: Decision per sample. Shape: (instances, )
    """
    rule_counts_probs = get_majority_vote_probs(rule_matches_z, mapping_rules_labels_t)

    def row_majority(row):
        row_max = np.max(row)
        num_occurrences = (row_max == row).sum()
        if num_occurrences == 1:
            return np.argmax(row)
        else:
            if no_rule_label is None:
                a = randint(0, rule_counts_probs.shape[1] - 1)
                if a >= 2:
                    print(a)
                    print(rule_counts_probs.shape)
                return a
            return no_rule_label

    majority_labels = np.apply_along_axis(row_majority, axis=1, arr=rule_counts_probs)
    return majority_labels


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


def get_majority_vote_probs_with_no_rel(
        rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray, no_match_class: int
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
    rule_counts[~rule_counts.any(axis=1), no_match_class] = 1
    rule_counts_probs = rule_counts / rule_counts.sum(axis=1).reshape(-1, 1)

    rule_counts_probs[np.isnan(rule_counts_probs)] = 0
    return rule_counts_probs
