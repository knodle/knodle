import logging
from typing import Tuple, Dict, Iterable

import numpy as np

logger = logging.getLogger(__name__)


def reduce_rule_matches(
    rule_matches_z: np.ndarray, mapping_rule_class_t: np.ndarray, drop_rules: bool = False,
    max_rules: int = None, min_coverage: float = None, rule_matches_rest: Dict[str, np.ndarray] = None
) -> Dict[str, np.ndarray]:

    """
    This is the main function to be used for reduction of rules.
    It finds the less effective rules by coverage in the train rule_matches_z and reduces them in the match matrix.
    The reduction is either done by dropping the rules (if drop_rules is True), or by creating one merged rule per label
    from the excessive rules selected for reduction (drop_rules is False).
    The mapping_rule_class_t matrix is adjusted accordingly. At least one of the criteria [max_rules, min_coverage]
    should be provided.
    Validation and test matches can be provided in rule_matches_rest to be reduced in the same manner as training data.

    Args:
        rule_matches_z: main match matrix, based on which the rules to keep are selected
        drop_rules: If True, the rules will be completely discarded, otherwise they will be
            represented as one rule (column) in reduced match matrix.
        max_rules: maximal number of rules to keep as-is. If drop_rules=False, with additional merged rules the total
            number of rules in the output may exceed this limit by #classes at most.
        min_coverage: minimal coverage required for a rule to be kept
        rule_matches_rest: additional rule_matches (dev, test) which should be reduced in the same manner
    Returns: Dictionary with "rule_matches_z", "mapping_rule_class_t" and keys from rule_matches_rest if provided.
    """
    if max_rules is None and min_coverage is None:
        logger.info("No filtering criteria ('max_rule' or 'min_coverage' for rule specified, "
                    "returning the original rule matches.")
        out = {"rule_matches_z": rule_matches_z, "mapping_rule_class_t": mapping_rule_class_t}
        out.update(rule_matches_rest)
        return out

    coverage_per_rule = rule_matches_z.sum(0) / rule_matches_z.shape[0]

    # create mask to indicate which rules will be kept unchanged
    rule_kept_mask = np.zeros(mapping_rule_class_t.shape[0], dtype=np.bool)

    # take top N rules
    if max_rules is not None:
        max_rule_ids = (coverage_per_rule * -1).argsort(kind="stable")[:max_rules]
        rule_kept_mask[max_rule_ids] = True

    # filter under-coveraged rules
    if min_coverage is not None:
        rule_kept_mask[coverage_per_rule < min_coverage] = False

    # add training matches to the dictionary
    rule_matches_dict = {"train_rule_matches_z": rule_matches_z}
    if rule_matches_rest is not None:
        rule_matches_dict.update(rule_matches_rest)

    if drop_rules:
        return _reduce_by_drop(rule_matches_dict, mapping_rule_class_t, rule_kept_mask)
    else:
        return _reduce_by_merge(rule_matches_dict, mapping_rule_class_t, rule_kept_mask)


def _reduce_by_drop(rule_matches_dict, mapping_rule_class_t, rule_kept_mask):
    output_dict = {}
    for split, match_matrix in rule_matches_dict.items():
        output_dict[split] = match_matrix[:, rule_kept_mask]
    output_dict["mapping_rule_class_t"] = mapping_rule_class_t[rule_kept_mask, :]
    return output_dict


def _reduce_by_merge(rule_matches_dict, mapping_rule_class_t, rule_kept_mask):
    # get core part of remaining rule matches
    output_dict = _reduce_by_drop(rule_matches_dict, mapping_rule_class_t, rule_kept_mask)

    # reduce all provided match matrices in the same manner; keep dict keys
    for split, full_match_matrix in rule_matches_dict.items():
        reduced_matches = _get_merged_matrix(
            full_matches=full_match_matrix,
            to_reduce_mask=~rule_kept_mask,
            label_rule_masks=_get_rule_by_label_iterator(mapping_rule_class_t)
        )

        # add merged rules to the core matches
        output_dict[split] = np.hstack([output_dict[split], reduced_matches])

    # add merged mapping
    merged_mapping = _get_merged_mapping(
        to_reduce_mask=~rule_kept_mask,
        label_rule_masks=_get_rule_by_label_iterator(mapping_rule_class_t),
        number_of_labels=mapping_rule_class_t.shape[1]
    )
    output_dict["mapping_rule_class_t"] = np.vstack([output_dict["mapping_rule_class_t"], merged_mapping])

    return output_dict


def _get_rule_by_label_iterator(
    mapping_rule_class_t: np.ndarray
) -> Iterable[np.ndarray]:
    """
    Get an iterator yielding an rule ids mask for every label, indicating which rules belong to this label.
    """
    for label_id in range(mapping_rule_class_t.shape[1]):
        column = mapping_rule_class_t[:, label_id]
        rule_mask = column != 0
        if mapping_rule_class_t[rule_mask].sum(0).nonzero()[0].tolist() != [label_id]:
            logger.warning(f"Rules for {label_id} point to multiple labels "
                           f"{mapping_rule_class_t[rule_mask].sum(0).nonzero()[0].tolist()}")
        yield rule_mask


def _get_merged_mapping(
    to_reduce_mask: np.ndarray, label_rule_masks: Iterable[np.ndarray], number_of_labels: int
) -> np.ndarray:
    """
    Creates a rule-to-class mapping matrix according to the rules selected for reduction.
    Returns: Mapping array of shape #merged rules (depends of #labels affected by reduction) x #labels
    """
    mappings_rule_class = []

    for label_id, label_mask in enumerate(label_rule_masks):
        to_reduce_label_mask = label_mask & to_reduce_mask

        # if there are no rules for this label that should be reduced, skip the label
        if not to_reduce_label_mask.any():
            continue

        mapping_label_row = np.zeros((number_of_labels,))

        # set the new rule to label with label_id
        mapping_label_row[label_id] = 1

        mappings_rule_class.append(mapping_label_row)

    return np.stack(mappings_rule_class, axis=0)


def _get_merged_matrix(
    full_matches: np.ndarray, to_reduce_mask: np.ndarray, label_rule_masks: Iterable[np.ndarray]
) -> np.ndarray:
    """
    Merge the selected columns from the original match matrix of x rules to a one-per-label rule.
    Args:
        full_matches: original match matrix with all rules: shape #obj x #rules
        to_reduce_mask: mask for the rules from 'full_matches' that should be reduced
        label_rule_masks: mapping of label indices to rule masks (shape: #rules)
    Returns: reduced matrix #obj x #labels
    """
    reduced_matches = []

    for label_id, label_mask in enumerate(label_rule_masks):
        # identify relevant rules from all label rules
        to_reduce_label_mask = label_mask & to_reduce_mask

        # if there are no rules for this label that should be reduced, skip the label
        if not to_reduce_label_mask.any():
            continue

        # create empty new rule column
        reduced_matches_col = np.zeros((full_matches.shape[0],))

        # the mask has to have *boolean type*, otherwise the matrix is sliced incorrectly
        label_rule_matches = full_matches[:, to_reduce_label_mask].sum(1)
        reduced_matches_col[label_rule_matches != 0] = 1  # identify matches where any original rule matched

        reduced_matches.append(reduced_matches_col)

    return np.stack(reduced_matches, axis=1)
