import logging
import random
import warnings
from typing import Tuple, Union

import numpy as np
import scipy.sparse as sp
from torch.utils.data import TensorDataset

from knodle.transformation.filter import filter_empty_probabilities, filter_probability_threshold

logger = logging.getLogger(__name__)


def input_to_majority_vote_input(
        rule_matches_z: np.ndarray,
        mapping_rules_labels_t: np.ndarray,
        model_input_x: TensorDataset = None,
        probability_threshold: int = None,
        unmatched_strategy: str = "random",  # other options: "filter", "other", "preserve"
        ties_strategy: str = "random",  # another options: "other"
        use_probabilistic_labels: bool = False,
        other_class_id: int = None,
        multi_label: bool = False,
        multi_label_threshold: float = None

) -> Union[Tuple[TensorDataset, TensorDataset, np.ndarray, np.ndarray], Tuple[TensorDataset, np.ndarray, np.ndarray]]:
    """
    This function converts the Knodle input (t and z matrices) into labels.

    The flow is the following:

    (1) the Z (samples x rules) and T (rules x classes) matrices are multiplied. The result of this multiplication is
        a matrix of a shape (samples x classes) with probabilities of every sample belonging to every class

    (2) if probability_threshold = True, the samples where probabilities for all classes are below the threshold value
        will be filtered out.

    (3) the samples where no labeling function matched (the samples where probabilities for all classes equal 0) will be
        handled in one of the following ways:
            - if unmatched_strategy = "filter", these samples will be filtered out.
            - if unmatched_strategy = "other", these samples will get the 100% probability for other class
            - if unmatched_strategy = "random", these samples will get the 100% probability for some random class
            - if unmatched_strategy = "preserve", these samples will remain unlabeled (with 0 probability vectors)
                WARNING! It is a valid use cases only if the probabilities will be used for training a classifier.
                If the probabilities are wished to be converted to majority labels, one of the first 3 handling
                strategies should be chosen.

    (4) if use_probabilistic_labels = False, the probabilities will be turned into labels with majority voting.
        That means:
            - if there is a clear "winner" (e.g. the probability of sample belonging to one class is clearly bigger
                than for the others), this class will be chosen as a final label for this sample
            - if there is a tie (e.g. probabilities of two or more classes are equal and larger than 0), it may be
                broken by:
                    - choosing a random class from the matched ones
                    - choosing an "other" label
    """
    check_input_validity(
        rule_matches_z,
        mapping_rules_labels_t,
        model_input_x,
        other_class_id,
        probability_threshold,
        unmatched_strategy,
        ties_strategy,
        use_probabilistic_labels
    )

    normalization = "sigmoid" if multi_label else "softmax"
    logger.info(f"{normalization} normalization will be used.")

    # (1) multiplication of Z and T matrices
    noisy_y_train = z_t_matrices_to_probs(rule_matches_z, mapping_rules_labels_t, normalization)

    # (2) filter out samples where all probabilities are below the threshold
    if probability_threshold is not None:
        model_input_x, noisy_y_train = filter_probability_threshold(model_input_x, noisy_y_train, probability_threshold)

    # (3) handling of samples without any match
    model_input_x, noisy_y_train, rule_matches_z = handle_non_labeled(
        model_input_x, noisy_y_train, rule_matches_z, unmatched_strategy=unmatched_strategy,
        other_class_id=other_class_id
    )

    # (4) turn probabilities into majority labels if relevant
    if not use_probabilistic_labels:
        # convert labels represented as a prob distribution to a single label using majority voting
        kwargs = {"ties_strategy": ties_strategy, "other_class_id": other_class_id}
        if multi_label:
            kwargs["threshold"] = multi_label_threshold
            noisy_y_train = np.apply_along_axis(probabilities_to_binary_multi_labels, axis=1, arr=noisy_y_train,
                                                **kwargs)
        else:
            noisy_y_train = np.apply_along_axis(probabilities_to_majority_vote, axis=1, arr=noisy_y_train, **kwargs)

    return model_input_x, noisy_y_train, rule_matches_z


def z_t_matrices_to_probs(
        rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray, normalization: str = "softmax"
) -> np.ndarray:
    """
    This function calculates a majority vote probability for all rule_matches_z_train.
    Args:
        rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
        mapping_rules_labels_t: Mapping of rules to labels, binary encoded. Shape: rules x classes
        normalization: how the rule counts will be normalized
    Returns: Array with majority vote probabilities. Shape: instances x classes
    """

    if isinstance(rule_matches_z, sp.csr_matrix):
        rule_counts = rule_matches_z.dot(mapping_rules_labels_t)
        if isinstance(rule_counts, sp.csr_matrix):
            rule_counts = rule_counts.toarray()
    else:
        rule_counts = np.matmul(rule_matches_z, mapping_rules_labels_t)

    if normalization == "softmax":
        rule_counts_probs = rule_counts / rule_counts.sum(axis=1).reshape(-1, 1)
    elif normalization == "sigmoid":
        rule_counts_probs = 1 / (1 + np.exp(-rule_counts))
        zeros = np.where(rule_counts == 0)  # the values that were 0s (= no LF from this class matched) should remain 0s
        rule_counts_probs[zeros] = rule_counts[zeros]
    else:
        raise ValueError(
            "Unknown label probabilities normalization; currently softmax and sigmoid normalization are supported"
        )
    rule_counts_probs[np.isnan(rule_counts_probs)] = 0
    return rule_counts_probs


def handle_non_labeled(
        input_data_x: TensorDataset,
        noisy_y_train: np.ndarray,
        rule_matches_z: np.ndarray = None,
        unmatched_strategy="random",
        other_class_id=None
) -> Tuple[TensorDataset, np.ndarray, np.ndarray]:
    """
    This function handles the samples where no labeling function matched (the samples where probabilities for all
    classes equal 0).
    Depending on the parameter values, these samples will be:
        - if unmatched_strategy = "filter", these samples will be filtered out.
        - if unmatched_strategy = "other", these samples will get the 100% probability for other class
        - if unmatched_strategy = "random", these samples will get the 100% probability for some random class
        - if unmatched_strategy = "preserve", these samples will remain unlabeled (with 0 probability vectors)
    """

    if len(noisy_y_train.shape) != 2:
        raise ValueError("noisy_y_train needs to be a matrix of dimensions num_samples x num_classes")

    prob_sums = noisy_y_train.sum(axis=-1)
    zeros = np.where(prob_sums == 0)[0]

    # filter these samples out
    if unmatched_strategy == "filter":
        logger.info("Samples with no labeling function matched will be filtered out")
        return filter_empty_probabilities(input_data_x, noisy_y_train, rule_matches_z)

    elif unmatched_strategy == "other":
        logger.info(f"Samples with no labeling function matched will be assigned to other class {other_class_id}")

        # other class id far from the maximal class -> change to the maximal class present in the data + 1
        if noisy_y_train.shape[1] < other_class_id:
            logger.info(f"The other class was changed {other_class_id} to {noisy_y_train.shape[1]} to match the data.")
            other_class_id = noisy_y_train.shape[1]

        # other class id = maximal class present in the data + 1
        if noisy_y_train.shape[1] == other_class_id:
            noisy_y_train = np.hstack((noisy_y_train, np.zeros([noisy_y_train.shape[0], 1])))
            noisy_y_train[zeros, other_class_id] = 1
        # other class id is already present in the data
        elif noisy_y_train.shape[1] > other_class_id:
            noisy_y_train[zeros, other_class_id] = 1
        else:
            raise ValueError("Invalid other class id!")

        return input_data_x, noisy_y_train, rule_matches_z

    elif unmatched_strategy == "random":
        logger.info("Samples with no labeling function matched will be assigned to random class")
        for sample_id in zeros:
            noisy_y_train[sample_id, np.random.choice(noisy_y_train.shape[1])] = 1
        return input_data_x, noisy_y_train, rule_matches_z

    elif unmatched_strategy == "preserve":
        logger.info("Samples with no labeling function matched will be preserved")
        return input_data_x, noisy_y_train, rule_matches_z


def probabilities_to_majority_vote(probs: np.ndarray, ties_strategy: str = "random", other_class_id: int = None) -> int:
    """Transforms a vector of probabilities to its majority vote. If there is one class with clear majority, return it.
    If there are more than one class with equal probabilities: either select one of the classes randomly or assign to
    the sample the other class id.
    Args:
        probs: Vector of probabilities for 1 sample. Shape: classes x 1
        ties_strategy: what to do with ties (choose a random label or an other class)
        other_class_id: Class ID being used, if there's no clear majority.
    Returns: An array of classes.
    """

    row_max = np.max(probs)
    num_occurrences = (row_max == probs).sum()
    if num_occurrences == 1:
        return int(np.argmax(probs))
    elif ties_strategy == "other":
        return other_class_id
    elif ties_strategy == "random":
        max_ids = np.where(probs == row_max)[0]
        return int(np.random.choice(max_ids))
    else:
        raise ValueError("Specify how to resolve unclear majority votes.")


def probabilities_to_binary_multi_labels(
        probs: np.ndarray, ties_strategy: str = "random", other_class_id: int = None, threshold: float = 0.5
) -> np.ndarray:
    """
    probs: Vector of probabilities for 1 sample. Shape: classes x 1
    choose_random_label: Choose a random label, if there's no clear majority.
    other_class_id: Class ID being used, if there's no clear majority
    threshold: a value for calculation the classes in case of multi-label classification: if a class has
        a probability greater than the threshold, this class will be selected as a true one
    """
    probs[probs >= threshold] = 1
    probs[probs < threshold] = 0

    if np.all((probs == 0)):
        if ties_strategy == "random":
            probs[:, random.randrange(probs.shape[1])] = 1
        elif ties_strategy == "other":
            probs[:, other_class_id] = 1
    return probs


def check_input_validity(*inputs) -> None:
    # todo: more checks are needed here?
    rule_matches_z, mapping_rules_labels_t, model_input_x, other_class_id, probability_threshold, \
    unmatched_strategy, ties_strategy, use_probabilistic_labels = inputs

    # check for other_class_id validity
    if other_class_id is not None and other_class_id < 0:
        raise RuntimeError("Label for negative samples should be greater than 0 for correct matrix multiplication")

    # check for matrices validity
    if rule_matches_z.shape[1] != mapping_rules_labels_t.shape[0]:
        raise ValueError(
            f"Dimensions mismatch! Z matrix has shape {rule_matches_z.shape}, "
            f"while T matrix has shape {mapping_rules_labels_t.shape}"
        )

    if other_class_id is not None and unmatched_strategy == "filter":
        raise ValueError("You can either filter samples with no weak labels or add them to the other class.")

    if unmatched_strategy == "random" is not None and other_class_id is not None:
        raise ValueError("You can either choose a random class, or transform undefined cases to an other class.")

    if (unmatched_strategy == "filter" and model_input_x is None) or \
            (probability_threshold is not None and model_input_x is None):
        raise ValueError("In order to filter non labeled samples, please provide X matrix.")

    if unmatched_strategy == "preserve" and not use_probabilistic_labels:
        raise ValueError("The empty probabilities cannot be preserved when calculating the majority labels.")

    if ties_strategy == "other" and other_class_id is None:
        raise ValueError("In order to break the ties by choosing an other class id, specify the other class id")

    if other_class_id is not None and other_class_id < mapping_rules_labels_t.shape[1] - 1:
        warnings.warn(f"Negative class {other_class_id} is already present in data")

    try:
        (unmatched_strategy in ["filter", "other", "preserve", "random"])
    except ValueError:
        logger.info("Unsupported strategy for dealing with samples without any rule matched.")

    if use_probabilistic_labels:
        try:
            (ties_strategy in ["other", "random"])
        except ValueError:
            logger.info("Unsupported strategy for dealing with samples without clear majority vote.")


def z_matrix_row_to_rule_idx(
        rules: np.ndarray, choose_random_rule: bool = False, multiple_instances: bool = False,
        no_match_class: int = -1
) -> int:
    """Transforms a z matrix row to rule indices of matching rules.
    If there is more than one rule match: either select one of the rules randomly or return a vector containing
    all of them.
    If there is no rule at all matching: assign the -1 class

    Args:
        rules: Vector of probabilities for 1 sample. Shape: classes x 1
        choose_random_rule: Choose a random label, if there's no clear majority.
        multiple_instances: Return duplicated instances with idx, if there are several rule matches.
        no_match_class: class that will be assigned if no LF matches (-1 class by default)
    Returns: An array of classes.
    """

    if choose_random_rule and multiple_instances:
        raise ValueError("You can either choose a random rule, or create multiple instances with multiple rules.")
    if isinstance(rules, sp.csr_matrix):  # TODO: how to do np.where with sparse matrices
        rules = rules.toarray()

    row_max = np.max(rules)
    num_occurrences = (row_max == rules).sum()
    if num_occurrences == 1:
        # there is only one matched LF
        return int(np.argmax(rules))
    else:
        max_ids = np.where(rules == row_max)[0]
        if row_max == 0.0 and len(max_ids) == len(rules):
            # if no LFs matched, assign no_match_class to the sample
            return no_match_class
        else:
            # if multiple LFs matched
            if choose_random_rule:
                # choose one LF randomly
                return int(np.random.choice(max_ids))
            elif multiple_instances:
                # select all LFs
                return np.where(rules == row_max)[0]
            else:
                raise ValueError("Specify a way how to resolve multiple rule matches.")


def z_t_matrices_to_labels_multi(
        rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray, choose_random_label_for_ties: bool,
        choose_other_label_for_ties: bool, other_class_id: int
) -> np.array:
    """Computes the majority labels. If no clear "winner" is found, other_class_id is used instead.
    Args:
        rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
        mapping_rules_labels_t: Mapping of rules to labels, binary encoded. Shape: rules x classes
        choose_random_label_for_ties: Whether a random label is chosen, if there's no clear majority vote.
        choose_other_label_for_ties: Whether an "other" label is chosen, if there's no clear majority vote.
        other_class_id: the id of other class, i.e. the class of negative samples
    Returns: Decision per sample. Shape: (instances, )
    """
    rule_counts_probs = z_t_matrices_to_probs_multi(rule_matches_z, mapping_rules_labels_t)

    kwargs = {
        "choose_random_label": choose_random_label_for_ties,
        "choose_other_label": choose_other_label_for_ties,
        "other_class_id": other_class_id
    }

    majority_labels = np.zeros((rule_counts_probs.shape[0], rule_counts_probs.shape[1], 1))
    for i, prob_row in enumerate(rule_counts_probs):
        mv_per_token = np.apply_along_axis(probabilities_to_majority_vote, axis=1, arr=prob_row, **kwargs)
        majority_labels[i] = np.reshape(mv_per_token, (-1, 1))
    return majority_labels


def z_t_matrices_to_probs_multi(
        rule_matches_z: np.ndarray,
        mapping_rules_labels_t: np.ndarray,
        normalization: str = "softmax"
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
        normalization: The way how the vectors will be normalized. Currently, there are two supported normalizations:
            - softmax
            - sigmoid
    Returns: Array with majority vote probabilities. Shape: instances x classes
    """
    rule_probs = np.zeros((rule_matches_z.shape[0], rule_matches_z.shape[1], mapping_rules_labels_t.shape[1]))
    for i, inst_row in enumerate(rule_matches_z):
        if inst_row.shape[1] != mapping_rules_labels_t.shape[0]:
            raise ValueError(f"Dimensions mismatch! Z matrix has shape {rule_matches_z.shape}, while "
                             f"T matrix has shape {mapping_rules_labels_t.shape}")

        if isinstance(inst_row, sp.csr_matrix):
            rule_counts = inst_row.dot(mapping_rules_labels_t)
            if isinstance(rule_counts, sp.csr_matrix):
                rule_counts = rule_counts.toarray()
        else:
            rule_counts = np.matmul(inst_row, mapping_rules_labels_t)

        if normalization == "softmax":
            rule_counts_probs = rule_counts / rule_counts.sum(axis=1).reshape(-1, 1)
        elif normalization == "sigmoid":
            rule_counts_probs = 1 / (1 + np.exp(-rule_counts))
            zeros = np.where(
                rule_counts == 0)  # the values that were 0s (= no LF from this class matched) should remain 0s
            rule_counts_probs[zeros] = rule_counts[zeros]
        else:
            raise ValueError(
                "Unknown label probabilities normalization; currently softmax and sigmoid normalization are supported"
            )
        rule_counts_probs[np.isnan(rule_counts_probs)] = 0
        rule_probs[i] = rule_counts_probs

    return rule_probs
