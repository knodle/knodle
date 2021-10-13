from typing import List

import numpy as np
from cleanlab.pruning import _prune_by_class, _prune_by_count
from cleanlab.util import value_counts, round_preserving_row_totals
from sklearn.preprocessing import normalize

# Leave at least this many examples in each class after
# pruning, regardless if noise estimates are larger.
MIN_NUM_PER_CLASS = 5         # 100


# def get_noise_indices(
#         noisy_labels,
#         psx,
#         confident_joint,
#         rule2class,
#         t_matrix,
#         prune_method='prune_by_noise_rate',
#         num_classes=None,
#         frac_noise=1.0,
#         num_to_remove_per_class=None,
#         multi_label=False,
# ) -> np.ndarray:
#     """
#
#     :param noisy_labels: original labels calculated from x and t multiplication
#     :param psx: probabilistic labels calculated with cross-validation
#     :param confident_joint: C matrix (rules x classes) with confident estimations
#     :param rule2class: t matrix = rules to classes correspondence
#     :param prune_method: prune method that will be used:
#             'prune_by_noise_rate':
#             'prune_by_class':
#             'both':
#             # todo: sorted_index_method
#     :param num_classes: number of output classes
#     :param frac_noise:
#     :param num_to_remove_per_class:
#     :param multi_label:  # todo: add multi_label
#     :return:
#     """
#     # todo: add multiprocessing as in the original CleanLab
#
#     # amount of samples in each class (according to the original noisy labels)
#     noisy_labels_counts = value_counts(noisy_labels)
#     noisy_labels = np.asarray(noisy_labels)
#
#     # todo: add custom_keep_at_least_n_per_class
#     # todo: add if num_to_remove_per_class is not None
#
#     args = (noisy_labels, noisy_labels_counts, confident_joint, psx, multi_label, t_matrix)
#
#
#     return t_matrix_updated


def update_t_matrix(prune_count_matrix, t_matrix, p=0.5) -> np.ndarray:
    normalized_prune_counts = normalize(prune_count_matrix, axis=1, norm='l1')
    updated_t_matrix = t_matrix * (1-p) + normalized_prune_counts * p

    # the first version: sum and divide
    # t_matrix_updated = np.zeros_like(t_matrix, dtype="float")
    # for rule in range(prune_count_matrix.shape[0]):
    #     rule_count_sum = sum(prune_count_matrix[rule, :]) + sum(t_matrix[rule, :])
    #     for label in range(prune_count_matrix.shape[1]):
    #         t_matrix_updated[rule][label] = (prune_count_matrix[rule][label] + t_matrix[rule][label]) / rule_count_sum

    return updated_t_matrix


def update_t_matrix_with_prior(prune_count_matrix, t_matrix) -> np.ndarray:
    """
    :param prune_count_matrix: C matrix (rules x classes) with confident estimations
    :return:
    """
    prior = np.mean(prune_count_matrix)
    print(f"Prior: {prior}")
    t_matrix_with_prior = t_matrix * prior
    updated_t_matrix = normalize(t_matrix_with_prior + prune_count_matrix, axis=1, norm='l1')
    return updated_t_matrix


def _prune_by_count_rule2class(k: int, args: List) -> np.ndarray:
    noisy_labels, noisy_labels_counts, prune_count_matrix, psx, multi_label = args
    prune_count_matrix = prune_count_matrix.T
    noise_mask = np.zeros(len(psx), dtype=bool)
    psx_k = psx[:, k]
    return noise_mask


"""
if args:  # Single processing - params are passed in
    s, s_counts, prune_count_matrix, psx, multi_label = args
else:  # Multiprocessing - data is shared across sub-processes
    s, s_counts, prune_count_matrix, psx, multi_label = _get_shared_data()

noise_mask = np.zeros(len(psx), dtype=bool)
psx_k = psx[:, k]
K = len(s_counts)
if s_counts[k] <= MIN_NUM_PER_CLASS:  # No prune if not MIN_NUM_PER_CLASS
    return np.zeros(len(s), dtype=bool)
for j in range(K):  # j is true label index (k is noisy label index)
    num2prune = prune_count_matrix[j][k]
    # Only prune for noise rates, not diagonal entries
    if k != j and num2prune > 0:
        # num2prune'th largest p(true class k) - p(noisy class k)
        # for x with true label j
        margin = psx[:, j] - psx_k
        s_filter = np.array(
            [k in lst for lst in s]
        ) if multi_label else s == k
        cut = -np.partition(-margin[s_filter], num2prune - 1)[num2prune - 1]
        noise_mask = noise_mask | (s_filter & (margin >= cut))
return noise_mask
"""


def custom_keep_at_least_n_per_class(
        prune_count_matrix: np.ndarray, mapping_rules_labels_t: np.ndarray, min_per_class: int, frac_noise: float = 1.0
) -> np.ndarray:
    """
    Make sure every class has at least n examples after removing noise.
    # Leave at least MIN_NUM_PER_CLASS examples per class.
    # NOTE prune_count_matrix is transposed (relative to confident_joint)
    """
    # get the number of samples corresponded to the original (noisy) t matrix entries (let's call them "equalities")
    # e.g. how many samples labeled originally with some label y' get the same label in prune_count_matrix
    prune_count_matrix_equals = prune_count_matrix.T[mapping_rules_labels_t.T == 1]

    # Set "equalities" that are less than n, to n.
    new_matrix_equals = np.maximum(prune_count_matrix_equals, min_per_class)

    # Find how much the "equalities" were increased.
    diff_per_rule = new_matrix_equals - prune_count_matrix_equals

    # Count non-zero items per rule (column), that are assigned not to the class this rule belongs to in t matrix
    # (i.e., that are not ""equalities").
    # np.maximum(*, 1) makes this never 0 (we divide by this next)
    num_noise_rates_per_rule = np.maximum(np.count_nonzero(prune_count_matrix, axis=0) - 1., 1., )

    # Uniformly decrease non-zero noise rates by the same amount
    # that the "equalities" were increased
    new_mat = prune_count_matrix - diff_per_rule / num_noise_rates_per_rule

    # Originally zero noise rates will now be negative, fix them back to zero
    new_mat[new_mat < 0] = 0

    # Round the "equalities" (correctly labeled examples)
    new_mat[mapping_rules_labels_t == 1] = new_matrix_equals

    # Reduce (multiply) all noise rates (non "equalities") by frac_noise and increase "equalities" by the total
    # amount reduced in each column to preserve column counts.
    new_mat = custom_reduce_prune_counts(new_mat, mapping_rules_labels_t, frac_noise)

    return round_preserving_row_totals(new_mat).astype(int)


def custom_reduce_prune_counts(
        prune_count_matrix: np.ndarray, mapping_rules_labels_t: np.ndarray, frac_noise: float = 1.0
) -> np.ndarray:
    # todo: clarify this function!
    eq_idx = (mapping_rules_labels_t == 1)

    # todo: this is relevant only if frac_noise != 1 -> clarify !
    new_mat = prune_count_matrix * frac_noise
    new_mat[eq_idx] = prune_count_matrix[eq_idx]

    diff = np.sum(prune_count_matrix - new_mat, axis=0)

    eq_idx_flat = eq_idx.flatten("F")
    out = eq_idx_flat.astype(diff.dtype)
    out[eq_idx_flat] = np.ravel(diff, "F")
    out = out.reshape(eq_idx.shape[0], eq_idx.shape[1], order='F')

    new_mat = np.add(new_mat, out)

    return new_mat.astype(int)
