import logging
from typing import Union, Dict, Iterable

import numpy as np
from scipy import sparse as ss

matrix = Union[np.ndarray, ss.csr_matrix]

logger = logging.getLogger(__name__)


def reduce_rule_matches(
    rule_matches_z: matrix, mapping_rules_labels_t: matrix, drop_rules: bool = False,
    max_rules: int = None, min_coverage: float = None, rule_matches_rest: Dict[str, matrix] = None
) -> Dict[str, matrix]:

    """
    This is the main function to be used for reduction of rules.
    It finds the less effective rules by coverage in the train rule_matches_z and reduces them in the match matrix.
    The reduction is either done by dropping the rules (if drop_rules is True), or by creating one merged rule per label
    from the excessive rules selected for reduction (drop_rules is False).
    The mapping_rules_labels_t matrix is adjusted accordingly. At least one of the criteria [max_rules, min_coverage]
    should be provided.
    Validation and test matches can be provided in rule_matches_rest to be reduced in the same manner as training data.

    Args:
        rule_matches_z: main match matrix, based on which the rules to keep are selected
        mapping_rules_labels_t: rule-to-label mapping, has to be reduced to the selected rules as well
        drop_rules: If True, the rules will be completely discarded, otherwise they will be
            represented as one rule (column) in reduced match matrix.
        max_rules: maximal number of rules to keep as-is. If drop_rules=False, with additional merged rules the total
            number of rules in the output may exceed this limit by #labels at most.
        min_coverage: minimal coverage required for a rule to be kept
        rule_matches_rest: additional rule_matches (dev, test) which should be reduced in the same manner
    Returns: Dictionary with "train_rule_matches_z", "mapping_rules_labels_t" and keys from rule_matches_rest if provided.
    """
    if max_rules is None and min_coverage is None:
        logger.info("No filtering criteria ('max_rule' or 'min_coverage' for rule specified, "
                    "returning the original rule matches.")
        out = {"train_rule_matches_z": rule_matches_z, "mapping_rules_labels_t": mapping_rules_labels_t}
        if rule_matches_rest:
            out.update(rule_matches_rest)
        return out

    rule_kept_ids = _select_rules_for_reduction(rule_matches_z, max_rules, min_coverage)

    # add training matches to the dictionary
    rule_matches_dict = {"train_rule_matches_z": rule_matches_z}
    if rule_matches_rest is not None:
        rule_matches_dict.update(rule_matches_rest)

    if drop_rules:
        return _reduce_by_drop(rule_matches_dict, mapping_rules_labels_t, rule_kept_ids)
    else:
        return _reduce_by_merge(rule_matches_dict, mapping_rules_labels_t, rule_kept_ids)


def _select_rules_for_reduction(
        rule_matches_z: matrix, max_rules: int = None, min_coverage: float = None
) -> np.ndarray:
    """
    Selects rule ids that match the reduction criteria based on rule coverage in the provided match matrix.
    At least one of the criteria [max_rules, min_coverage] should be provided.
    Args:
        rule_matches_z: main match matrix, based on which the rules to keep are selected
        max_rules: maximal number of rules to keep as-is.
        min_coverage: minimal coverage required for a rule to be kept
    Returns: Array with rule ids.
    """
    coverage_per_rule = rule_matches_z.sum(0) / rule_matches_z.shape[0]

    if isinstance(rule_matches_z, ss.csr_matrix):
        # convert from numpy.matrix (result of sparse sum) to numpy.array
        coverage_per_rule = np.array(coverage_per_rule).flatten()

    # create a set to collect ids of rules which will be kept unchanged
    rule_kept_ids = np.array(range(rule_matches_z.shape[1]))

    # take top N rules
    if max_rules is not None:
        max_rule_ids = (coverage_per_rule * -1).argsort(kind="stable")[:max_rules]
        rule_kept_ids = np.intersect1d(rule_kept_ids, max_rule_ids)

    # filter under-coveraged rules
    if min_coverage is not None:
        min_coverage_ids = np.nonzero(coverage_per_rule < min_coverage)[0]
        rule_kept_ids = np.setdiff1d(rule_kept_ids, min_coverage_ids)

    # convert to array to slice matrices
    return rule_kept_ids


def _reduce_by_drop(
        rule_matches_dict: Dict[str, matrix], mapping_rules_labels_t: matrix, rule_kept_ids: np.ndarray
) -> Dict[str, matrix]:
    """
    Drops the rules according to the provided mask. Performs same reduction on all match matrices in the dictionary.
    Corresponding columns of mapping matrix T are dropped as well.
    If N rules were selected to *not* be reduced (there are N entries in the `rule_kept_ids`):
    Args:
        rule_matches_dict: contains the rule match matrices (#objects x N)
        mapping_rules_labels_t: rule-to-class mapping (N x #labels)
        rule_kept_ids: ids of rules that should *not* be reduced (N)

    Returns: Reduced match matrices for all keys in the `rule_matches_dict`, reduced mapping T in a single dictionary.
    """
    output_dict = {}
    for split, match_matrix in rule_matches_dict.items():
        output_dict[split] = match_matrix[:, rule_kept_ids]
    output_dict["mapping_rules_labels_t"] = mapping_rules_labels_t[rule_kept_ids, :]
    return output_dict


def _reduce_by_merge(
        rule_matches_dict: Dict[str, matrix], mapping_rules_labels_t: matrix, rule_kept_ids: np.ndarray
) -> Dict[str, matrix]:
    """
    Leaves rules selected according to the provided mask unchanged
    while merging the others based on their corresponding labels.
    Performs same reduction on all match matrices in the dictionary.
    Corresponding columns of mapping matrix T are processed similarly and will contain an unchanged and merged columns.
    If N rules were selected to *not* be reduced (there are N entries in the `rule_kept_ids`),
    and the other rules correspond to M labels, the resulting set of rules will contain N + M rules,
    Args:
        rule_matches_dict: contains the rule match matrices (#objects x (N + M))
        mapping_rules_labels_t: rule-to-label mapping ((N + M) x #labels)
        rule_kept_ids: ids of rules that should *not* be reduced (N)

    Returns: Reduced match matrices for all keys in the `rule_matches_dict`, reduced mapping T in a single dictionary.
    """

    # get core part of remaining rule matches
    output_dict = _reduce_by_drop(rule_matches_dict, mapping_rules_labels_t, rule_kept_ids)

    # reduce all provided match matrices in the same manner; keep dict keys
    for split, full_match_matrix in rule_matches_dict.items():
        reduced_matches = _get_merged_matrix(
            full_matches=full_match_matrix,
            to_keep_mask=rule_kept_ids,
            label_rule_masks=_get_rule_by_label_iterator(mapping_rules_labels_t)
        )

        # add merged rules to the core matches
        if isinstance(output_dict[split], ss.csr_matrix):
            # add columns with sparse columns, then convert back to rows
            output_dict[split] = ss.hstack([output_dict[split].tocsc(), reduced_matches]).tocsr()
        else:
            output_dict[split] = np.hstack([output_dict[split], reduced_matches])

    # add merged mapping
    merged_mapping = _get_merged_mapping(
        to_keep_ids=rule_kept_ids,
        label_rule_masks=_get_rule_by_label_iterator(mapping_rules_labels_t),
        number_of_labels=mapping_rules_labels_t.shape[1],
        make_sparse=isinstance(mapping_rules_labels_t, ss.csr_matrix)
    )

    if isinstance(merged_mapping, ss.csr_matrix):
        output_dict["mapping_rules_labels_t"] = ss.vstack([output_dict["mapping_rules_labels_t"], merged_mapping])
    else:
        output_dict["mapping_rules_labels_t"] = np.vstack([output_dict["mapping_rules_labels_t"], merged_mapping])

    return output_dict


def _get_rule_by_label_iterator(
    mapping_rules_labels_t: matrix
) -> Iterable[np.ndarray]:
    """
    Get an iterator yielding an rule ids mask for every label, indicating which rules belong to this label.
    """
    for label_id in range(mapping_rules_labels_t.shape[1]):
        column = mapping_rules_labels_t[:, label_id]
        rule_mask = column.nonzero()[0]

        # check if rules for this label also correspond to other labels
        if isinstance(mapping_rules_labels_t, ss.csr_matrix):
            labels_for_rules = mapping_rules_labels_t[rule_mask].sum(0).nonzero()[1].tolist()
        else:
            labels_for_rules = mapping_rules_labels_t[rule_mask].sum(0).nonzero()[0].tolist()

        if labels_for_rules != [label_id]:
            logger.warning(f"Rules for {label_id} point to multiple labels {labels_for_rules}")
        yield rule_mask


def _get_merged_mapping(
    to_keep_ids: np.ndarray, label_rule_masks: Iterable[np.ndarray], number_of_labels: int, make_sparse: bool
) -> np.ndarray:
    """
    Creates a rule-to-label mapping matrix according to the rules selected for reduction.
    Returns: Mapping array of shape M (depends of #labels affected by reduction) x #labels
    """
    mappings_rules_labels = []

    for label_id, label_mask in enumerate(label_rule_masks):
        #to_reduce_label_mask = label_mask & to_reduce_mask
        to_reduce_label_mask = np.setdiff1d(label_mask, to_keep_ids)

        # if there are no rules for this label that should be reduced, skip the label
        if len(to_reduce_label_mask) == 0:
            continue

        mapping_label_row = np.zeros((number_of_labels,))

        # set the new rule to label with label_id
        mapping_label_row[label_id] = 1

        if make_sparse:
            mapping_label_row = ss.csr_matrix(mapping_label_row)

        mappings_rules_labels.append(mapping_label_row)

    if make_sparse:
        return ss.vstack(mappings_rules_labels)
    else:
        return np.vstack(mappings_rules_labels)


def _get_merged_matrix(
    full_matches: matrix, to_keep_mask: np.ndarray, label_rule_masks: Iterable[np.ndarray]
) -> matrix:
    """
    Merge the selected columns from the original match matrix of x rules to a one-per-label rule.
    If a label corresponds to any of the rules selected for reduction, a new merged rule is constructed for this label
    which contains matches from all to-be-reduced rules for this label.

    Args:
        full_matches: original match matrix with all rules: shape #obj x #rules
        to_keep_mask: mask for the rules from 'full_matches' that should be reduced
        label_rule_masks: mapping of label indices to rule masks (shape: #rules)
    Returns: reduced matrix #obj x #labels
    """
    reduced_matches = []

    for label_id, label_mask in enumerate(label_rule_masks):
        # identify relevant rules from all label rules
        to_reduce_label_mask = np.setdiff1d(label_mask, to_keep_mask)

        # if there are no rules for this label that should be reduced, skip the label
        if len(to_reduce_label_mask) == 0:
            continue

        # create a merged match column
        reduced_matches_col = full_matches[:, to_reduce_label_mask].sum(1)

        # take care of multiple matches per row --> reduce them to one
        if isinstance(full_matches, ss.csr_matrix):
            # create sparse *column* for faster column-wise rule match processing
            reduced_matches_col = ss.csc_matrix(reduced_matches_col).minimum(1)
        else:
            reduced_matches_col = np.minimum(reduced_matches_col, 1).reshape(-1, 1)

        reduced_matches.append(reduced_matches_col)

    if isinstance(full_matches, ss.csr_matrix):
        return ss.hstack(reduced_matches)
    else:
        return np.hstack(reduced_matches)
