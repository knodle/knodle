from typing import Tuple

import numpy as np
import scipy.sparse as sp
from torch.utils.data import TensorDataset

from knodle.transformation.filter import filter_tensor_dataset_by_indices


def z_t_matrix_to_snorkel_matrix(rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray) -> np.ndarray:
    snorkel_matrix = -1 * np.ones(rule_matches_z.shape)

    if isinstance(rule_matches_z, sp.csr_matrix):
        rule_matches_z = rule_matches_z.toarray()

    z_to_t = np.argmax(mapping_rules_labels_t, axis=-1)
    for i in range(rule_matches_z.shape[0]):
        non_zero_idx = np.where(rule_matches_z[i] != 0)[0]
        snorkel_matrix[i, non_zero_idx] = z_to_t[non_zero_idx]

    return snorkel_matrix

def prepare_empty_rule_matches(rule_matches_z, filter_non_labelled, model_input_x
) -> Tuple[np.ndarray, np.ndarray, TensorDataset]:
    """
    Remove empty rows in rule matches for LabelModel. If filtering is configured, the corresponding entries
    in the model input are filtered out as well.
    Args:
        rule_matches_z:
        filter_non_labelled: if rows with no rule matches should be filtered.
        model_input_x:
    Returns:
        boolean mask indicating non-empty rows,
        filtered rule matches,
        and eventually filtered model input
    """
    # find empty rows of the rule matches
    non_zero_mask = rule_matches_z.sum(axis=1) != 0
    non_zero_indices = np.where(non_zero_mask)[0]

    # exclude empty rows from LabelModel input
    rule_matches_z = rule_matches_z[non_zero_indices]

    if filter_non_labelled:
        # filter out respective input irrevocably from all data
        model_input_x = filter_tensor_dataset_by_indices(dataset=model_input_x, filter_ids=non_zero_indices)

    return non_zero_mask, rule_matches_z, model_input_x

def add_labels_for_empty_examples(label_probs_gen, non_zero_mask, output_classes, other_class_id
) -> np.ndarray:
    """
    Args:
        label_probs_gen: labels generated for the non-empty rows
        non_zero_mask:  boolean mask indicating which examples get the generated labels
        output_classes: number of output classes, with respect to the other_class
        other_class_id: id of the class for empty rows
    Returns: distribution of labels for both empty and non-empty rows (#examples x #classes)
    """
    # make dummy label distibutions for all of the examples
    # number of output classes is eventually t.shape[1]+1, if other class id should be added
    label_probs = np.zeros((non_zero_mask.shape[0], output_classes))

    # fill labels of the generative model
    label_probs[non_zero_mask, :label_probs_gen.shape[1]] = label_probs_gen

    # assign full probability to other class for empty rows
    label_probs[~non_zero_mask, other_class_id] = 1
    return label_probs
