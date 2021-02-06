import numpy as np
from torch.utils.data import TensorDataset

from knodle.transformation.filter import filter_empty_probabilities


def probabilies_to_majority_vote(
        probs: np.ndarray, choose_random_label: bool = True, other_class_id: int = None
) -> int:
    """Tranforms a vector of probabilites to its majority vote.

    :param probs: Vector of probabilites for 1 sample
    :param choose_random_label: Choose a random label, if there's no clear majority. If there exists a
        probability > 0, takes preference over other_class_id, otherwise other_class_id
    :param other_class_id: Class ID being used, if there's no clear majority
    :return:
    """
    if choose_random_label and other_class_id is not None:
        raise ValueError("You can either choose a random class, or transform undefined cases to an other class.")

    row_max = np.max(probs)
    num_occurrences = (row_max == probs).sum()
    if num_occurrences == 1:
        return int(np.argmax(probs))
    elif choose_random_label:
        max_ids = np.where(probs == row_max)[0]
        return int(np.random.choice(max_ids))
    elif other_class_id is not None:
        return other_class_id
    else:
        raise ValueError("Specify a way how to resolve unclear majority votes.")


def z_t_matrices_to_majority_vote_probs(
        rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray, other_class: int = None
) -> np.ndarray:
    """
    This function calculates a majority vote probability for all rule_matches_z. The difference from simple
    get_majority_vote_probs function is the following: samples, where no rules matched (that is, all elements in
    the corresponding raw in rule_matches_z matrix equal 0), are assigned to no_match_class (that is, a value in the
    corresponding column in rule_counts_probs matrix is changed to 1).

    Args:
        rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
        mapping_rules_labels_t: Mapping of rules to labels, binary encoded. Shape: rules x classes
        other_class: Class which is chosen, if no function is hitting.
    Returns: Array with majority vote probabilities. Shape: instances x classes
    """
    if rule_matches_z.shape[1] != mapping_rules_labels_t.shape[0]:
        raise ValueError("Dimensions mismatch!")

    rule_counts = np.matmul(rule_matches_z, mapping_rules_labels_t)
    if other_class is not None:
        rule_counts[~rule_counts.any(axis=1), other_class] = 1
    rule_counts_probs = rule_counts / rule_counts.sum(axis=1).reshape(-1, 1)

    rule_counts_probs[np.isnan(rule_counts_probs)] = 0
    return rule_counts_probs


def z_t_matrices_to_majority_vote_labels(
        rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray,
        choose_random_label: bool = True, other_class_id: int = None
) -> np.array:
    """Computes the majority labels. If no clear "winner" is found, no_rule_label is used instead.
    Args:
        rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
        mapping_rules_labels_t: Mapping of rules to labels, binary encoded. Shape: rules x classes
        choose_random_label: Whether a random label is chosen, if there's no clear majority vote.
        other_class_id: Dummy label
    Returns: Decision per sample. Shape: (instances, )
    """
    rule_counts_probs = z_t_matrices_to_majority_vote_probs(rule_matches_z, mapping_rules_labels_t)

    kwargs = {"choose_random_label": choose_random_label, "other_class_id": other_class_id}
    majority_labels = np.apply_along_axis(probabilies_to_majority_vote, axis=1, arr=rule_counts_probs, **kwargs)
    return majority_labels


def input_to_majority_vote_input(
        input_data_x: TensorDataset, rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray,
        filter_empty_z_rows: bool = True, other_class_id: int = None, return_probs: bool = True
):
    if other_class_id is not None and filter_empty_z_rows:
        raise ValueError("You can either filter samples with no weak labels or add them to 'other_class_id'")

    label_probs = z_t_matrices_to_majority_vote_probs(rule_matches_z, mapping_rules_labels_t, other_class_id)
    if filter_empty_z_rows:
        input_data_x, label_probs = filter_empty_probabilities(input_data_x, label_probs)

    if not return_probs:
        kwargs = {"choose_random_label": True, "other_class_id": other_class_id}
        label_probs = np.apply_along_axis(probabilies_to_majority_vote, axis=1, arr=label_probs, **kwargs)

    return input_data_x, label_probs
