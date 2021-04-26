from typing import Tuple, Union

import numpy as np
from torch.utils.data import TensorDataset


def filter_empty_probabilities(
        input_data_x: TensorDataset, class_probas_y: np.ndarray, rule_matches_z: np.ndarray = None
) -> Union[Tuple[TensorDataset, np.ndarray, np.ndarray], Tuple[TensorDataset, np.ndarray]]:
    """Delete rows of TensorDataset's where the cumulative probability equals 0.
    Args:
        input_data_x: A TensorDataset serving as input to a model
        class_probas_y: Array, holding class probabilities, shape=num_samples, num_classes
        rule_matches_z: optional array with rules matched in samples. If given, will also be filtered
    :return: Modified TensorDataset's
    """
    if len(class_probas_y.shape) != 2:
        raise ValueError("y_probs needs to be a matrix of dimensions num_samples x num_classes")

    prob_sums = class_probas_y.sum(axis=-1)
    non_zeros = np.where(prob_sums != 0)[0]

    new_tensors = []
    for i in range(len(input_data_x.tensors)):
        new_tensors.append(input_data_x.tensors[i][non_zeros])
    new_x = TensorDataset(*new_tensors)

    if rule_matches_z is not None:
        return new_x, class_probas_y[non_zeros], rule_matches_z[non_zeros]

    return new_x, class_probas_y[non_zeros]


def filter_unconclusive_probabilities(
        input_data_x: TensorDataset, class_probas_y: np.ndarray, rule_matches_z: np.ndarray = None,
        probability_threshold: float = 0.7
) -> Union[Tuple[TensorDataset, np.ndarray, np.ndarray], Tuple[TensorDataset, np.ndarray]]:
    """Filters instances where no single class probability exceeds "probability_threshold".
    """
    prob_sums = class_probas_y.max(axis=-1)
    conclusive_idx = np.where(prob_sums >= probability_threshold)[0]

    new_tensors = []
    for i in range(len(input_data_x.tensors)):
        new_tensors.append(input_data_x.tensors[i][conclusive_idx])
    new_x = TensorDataset(*new_tensors)

    if rule_matches_z is not None:
        return new_x, class_probas_y[conclusive_idx], rule_matches_z[conclusive_idx]

    return new_x, class_probas_y[conclusive_idx]
