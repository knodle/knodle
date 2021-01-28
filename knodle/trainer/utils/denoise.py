import numpy as np
from tqdm import tqdm


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


# def activate_all_neighbors(
#         rule_matches_z: np.ndarray, indices: np.ndarray
# ) -> np.ndarray:
#     """
#     Find all closest neighbors and take the same label ids
#     Args:
#         rule_matches_z: All rule matches. Shape: instances x rules
#         indices: Neighbor indices from knn
#     Returns:
#     """
#     new_lfs_array = np.full(rule_matches_z.shape, fill_value=0)
#
#     for index, lf in enumerate(rule_matches_z):
#
#         try:
#             matched_lfs = np.where(lf != 0)[0]
#             if len(matched_lfs) == 0:
#                 continue
#             matched_lfs = matched_lfs[:, np.newaxis]
#             neighbors = indices[index]
#             to_replace = new_lfs_array[neighbors, matched_lfs]
#             label_matched_lfs = lf[matched_lfs][:, 0]
#             tiled_labels = np.tile(
#                 np.array(label_matched_lfs), (to_replace.shape[1], 1)
#             ).transpose()
#             new_lfs_array[neighbors, matched_lfs] = tiled_labels
#         except IndexError:
#             pass
#
#     return new_lfs_array


def activate_all_neighbors(
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
