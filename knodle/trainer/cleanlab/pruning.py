from typing import List

import numpy as np
from cleanlab.pruning import _prune_by_class, _prune_by_count
from cleanlab.util import value_counts, round_preserving_row_totals

# Leave at least this many examples in each class after
# pruning, regardless if noise estimates are larger.
MIN_NUM_PER_CLASS = 5         # 100


def get_noise_indices(
        noisy_labels,
        psx,
        confident_joint,
        rule2class,
        prune_method='prune_by_noise_rate',
        num_classes=None,
        frac_noise=1.0,
        num_to_remove_per_class=None,
        multi_label=False,
) -> np.ndarray:
    # todo: add multiprocessing as in original CleanLab
    # todo: add multi_label
    # todo: sorted_index_method

    noisy_labels_counts = value_counts(noisy_labels)
    noisy_labels = np.asarray(noisy_labels)  # ensure labels are of type np.array()

    # Number of classes s
    if not num_classes:
        num_classes = len(psx.T)

    # Leave at least MIN_NUM_PER_CLASS examples per class.
    # NOTE prune_count_matrix is transposed (relative to confident_joint)
    prune_count_matrix = custom_keep_at_least_n_per_class(
        prune_count_matrix=confident_joint.T,
        mapping_rules_labels_t=rule2class.T,
        min_per_class=MIN_NUM_PER_CLASS,
        frac_noise=frac_noise,
    )

    if num_to_remove_per_class is not None:
        # Estimate joint probability distribution over label errors
        psy = prune_count_matrix / np.sum(prune_count_matrix, axis=1)
        noise_per_noisy_label = psy.sum(axis=1) - psy.diagonal()
        # Calibrate s.t. noise rates sum to num_to_remove_per_class
        tmp = (psy.T * num_to_remove_per_class / noise_per_noisy_label).T
        np.fill_diagonal(tmp, noisy_labels_counts - num_to_remove_per_class)
        prune_count_matrix = round_preserving_row_totals(tmp)

    args = (noisy_labels, noisy_labels_counts, prune_count_matrix, psx, multi_label)

    # Perform Pruning with threshold probabilities from BFPRT algorithm in O(n)
    if prune_method == 'prune_by_class' or prune_method == 'both':
        raise ValueError("Method is not implemented yet.")

    if prune_method == 'prune_by_noise_rate':
        noise_masks_per_class = [_prune_by_count(k, args) for k in range(num_classes)]
        label_errors_mask = np.stack(noise_masks_per_class).any(axis=0)

    # Remove label errors if given label == model prediction
    pred = psx.argmax(axis=1)
    for i, pred_label in enumerate(pred):
        if pred_label == noisy_labels[i]:
            label_errors_mask[i] = False

    return label_errors_mask


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

    # get the entries which are corresponded to the original (noisy) t matrix entries (let's call them "equalities")
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

    eq_idx = (mapping_rules_labels_t == 1)

    new_mat = prune_count_matrix * frac_noise
    new_mat[eq_idx] = prune_count_matrix[eq_idx]

    diff = np.sum(prune_count_matrix - new_mat, axis=0)

    eq_idx_flat = eq_idx.flatten("F")
    out = eq_idx_flat.astype(diff.dtype)
    out[eq_idx_flat] = np.ravel(diff, "F")
    out = out.reshape(eq_idx.shape[0], eq_idx.shape[1], order='F')

    new_mat = np.add(new_mat, out)

    return new_mat.astype(int)
