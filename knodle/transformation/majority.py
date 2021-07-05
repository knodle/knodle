import numpy as np
import scipy.sparse as sp
import warnings

from torch.utils.data import TensorDataset

from knodle.transformation.filter import filter_empty_probabilities, filter_probability_threshold


def probabilies_to_majority_vote(
        probs: np.ndarray, choose_random_label: bool = True, other_class_id: int = None
) -> int:
    """Transforms a vector of probabilities to its majority vote. If there is one class with clear majority, return it.
    If there are more than one class with equal probabilities: either select one of the classes randomly or assign to
    the sample the other class id.
 
    Args:
        probs: Vector of probabilities for 1 sample. Shape: classes x 1
        choose_random_label: Choose a random label, if there's no clear majority.
        other_class_id: Class ID being used, if there's no clear majority
    Returns: An array of classes.
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
        rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray, other_class_id: int = None
) -> np.ndarray:
    """
    This function calculates a majority vote probability for all rule_matches_z. The difference from simple
    get_majority_vote_probs function is the following: samples, where no rules matched (that is, all elements in
    the corresponding raw in rule_matches_z matrix equal 0), are assigned to no_match_class (that is, a value in the
    corresponding column in rule_counts_probs matrix is changed to 1).

    Args:
        rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
        mapping_rules_labels_t: Mapping of rules to labels, binary encoded. Shape: rules x classes
        other_class_id: Class which is chosen, if no function is hitting.
    Returns: Array with majority vote probabilities. Shape: instances x classes
    """

    if rule_matches_z.shape[1] != mapping_rules_labels_t.shape[0]:
        raise ValueError(f"Dimensions mismatch! Z matrix has shape {rule_matches_z.shape}, while "
                         f"T matrix has shape {mapping_rules_labels_t.shape}")

    if isinstance(rule_matches_z, sp.csr_matrix):
        rule_counts = rule_matches_z.dot(mapping_rules_labels_t)
        if isinstance(rule_counts, sp.csr_matrix):
            rule_counts = rule_counts.toarray()
    else:
        rule_counts = np.matmul(rule_matches_z, mapping_rules_labels_t)

    if other_class_id:
        if other_class_id < 0:
            raise RuntimeError("Label for negative samples should be greater than 0 for correct matrix multiplication")
        if other_class_id < mapping_rules_labels_t.shape[1] - 1:
            warnings.warn(f"Negative class {other_class_id} is already present in data")
        if rule_counts.shape[1] == other_class_id:
            rule_counts = np.hstack((rule_counts, np.zeros([rule_counts.shape[0], 1])))
            rule_counts[~rule_counts.any(axis=1), other_class_id] = 1
        elif rule_counts.shape[1] >= other_class_id:
            rule_counts[~rule_counts.any(axis=1), other_class_id] = 1
        else:
            raise ValueError("Other class id is incorrect")
    rule_counts_probs = rule_counts / rule_counts.sum(axis=1).reshape(-1, 1)
    rule_counts_probs[np.isnan(rule_counts_probs)] = 0
    return rule_counts_probs


def z_t_matrices_to_majority_vote_labels(
        rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray,
        choose_random_label: bool = True, other_class_id: int = None
) -> np.array:
    """Computes the majority labels. If no clear "winner" is found, other_class_id is used instead.
    Args:
        rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
        mapping_rules_labels_t: Mapping of rules to labels, binary encoded. Shape: rules x classes
        choose_random_label: Whether a random label is chosen, if there's no clear majority vote.
        other_class_id: the id of other class, i.e. the class of negative samples
    Returns: Decision per sample. Shape: (instances, )
    """
    rule_counts_probs = z_t_matrices_to_majority_vote_probs(rule_matches_z, mapping_rules_labels_t)

    kwargs = {"choose_random_label": choose_random_label, "other_class_id": other_class_id}
    majority_labels = np.apply_along_axis(probabilies_to_majority_vote, axis=1, arr=rule_counts_probs, **kwargs)
    return majority_labels


def input_to_majority_vote_input(
        rule_matches_z: np.ndarray,
        mapping_rules_labels_t: np.ndarray,
        model_input_x: TensorDataset,
        use_probabilistic_labels: bool = True,
        filter_non_labelled: bool = True,
        probability_threshold: int = None,
        other_class_id: int = None,
) -> np.ndarray:
    """
    This function calculates noisy labels y_hat from Knodle Z and T matrices.
    :param model_input_x:
    :param rule_matches_z: binary encoded array of which rules matched. Shape: instances x rules
    :param mapping_rules_labels_t: mapping of rules to labels, binary encoded. Shape: rules x classes
    :param filter_non_labelled: boolean value, whether the no matched samples should be filtered out or not.
    :param other_class_id: the id of other class, i.e. the class of no matched samples, if they are to be stored.
    :param use_probabilistic_labels: boolean value, whether the output labels should be in form of probabilistic labels
    or single values.
    :return:
    """
    if other_class_id is not None and filter_non_labelled:
        raise ValueError("You can either filter samples with no weak labels or add them to the other class.")

    noisy_y_train = z_t_matrices_to_majority_vote_probs(rule_matches_z, mapping_rules_labels_t, other_class_id)

    if filter_non_labelled and probability_threshold is not None:
        raise ValueError("You can either filter all non labeled samples or those that have probabilities below "
                         "some threshold.")

    #  filter out samples where no pattern matched
    if filter_non_labelled:
        model_input_x, noisy_y_train, rule_matches_z = filter_empty_probabilities(
            model_input_x, noisy_y_train, rule_matches_z
        )

    #  filter out samples where that have probabilities below the threshold
    elif probability_threshold is not None:
        model_input_x, noisy_y_train = filter_probability_threshold(
            model_input_x, noisy_y_train, probability_threshold=probability_threshold
            )

    if not use_probabilistic_labels:
        # convert labels represented as a prob distribution to a single label using majority voting
        kwargs = {"choose_random_label": True, "other_class_id": other_class_id}
        noisy_y_train = np.apply_along_axis(probabilies_to_majority_vote, axis=1, arr=noisy_y_train, **kwargs)

    return model_input_x, noisy_y_train, rule_matches_z
