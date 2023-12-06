from typing import Tuple, Union, List

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import Subset


def filter_tensor_dataset_by_indices(dataset: TensorDataset, filter_ids: Union[np.ndarray, List[int]]) -> TensorDataset:
    """Filters each tensor of a TensorDataset, given some "filter_ids".

    Args:
        dataset: TensorDataset with a list of tensors, each having first dimension N
        filter_ids: A list of K indices to be kept, K <= N

    Returns: TensorDataset with filtered indices
    """
    new_tensors = []
    for i in range(len(dataset.tensors)):
        new_tensors.append(dataset.tensors[i][filter_ids])
    dataset_new = TensorDataset(*new_tensors)

    return dataset_new


def filter_empty_probabilities(
        input_data_x: TensorDataset, class_probas_y: np.ndarray, rule_matches_z: np.ndarray = None
) -> Union[Tuple[TensorDataset, np.ndarray, np.ndarray], Tuple[TensorDataset, np.ndarray]]:
    """Delete rows of TensorDataset's where the cumulative probability equals 0.
    Args:
        input_data_x: a TensorDataset serving as input to a model
        class_probas_y: array, holding class probabilities, shape=num_samples, num_classes
        rule_matches_z_train: optional array with rules matched in samples. If given, will also be filtered
        add_input_data_x: additional TensorDataset that has the same dimension as the input_data_x and that also should
            be filtered (used in trainers where the same input data is encoded with different features - e.g. TF-IDF
            for the denoising and BERT for the main classifier training - but should be filtered in the same way)
    :return: Modified TensorDataset's
    """
    if len(class_probas_y.shape) != 2:
        raise ValueError("y_probs needs to be a matrix of dimensions num_samples x num_classes")

    prob_sums = class_probas_y.sum(axis=-1)
    non_zeros = np.where(prob_sums != 0)[0]

    new_x = filter_tensor_dataset_by_indices(dataset=input_data_x, filter_ids=non_zeros)

    if rule_matches_z is not None:
        return new_x, class_probas_y[non_zeros], rule_matches_z[non_zeros]

    return new_x, class_probas_y[non_zeros]


def filter_empty_probabilities_two_datasets(
        input_data_x: TensorDataset, add_input_data_x: TensorDataset, class_probas_y: np.ndarray,
        rule_matches_z: np.ndarray = None
) -> Union[Tuple[TensorDataset, TensorDataset, np.ndarray, np.ndarray], Tuple[TensorDataset, TensorDataset, np.ndarray]]:
    """Delete rows of TensorDataset's where the cumulative probability equals 0.
    Args:
        input_data_x: a TensorDataset serving as input to a model
        add_input_data_x: additional TensorDataset that has the same dimension as the input_data_x and that also should
            be filtered (used in trainers where the same input data is encoded with different features - e.g. TF-IDF
            for the denoising and BERT for the main classifier training - but should be filtered in the same way)
        class_probas_y: array, holding class probabilities, shape=num_samples, num_classes
        rule_matches_z_train: optional array with rules matched in samples. If given, will also be filtered
    :return: Modified TensorDataset's
    """
    if len(class_probas_y.shape) != 2:
        raise ValueError("y_probs needs to be a matrix of dimensions num_samples x num_classes")

    prob_sums = class_probas_y.sum(axis=-1)
    non_zeros = np.where(prob_sums != 0)[0]

    new_x = filter_tensor_dataset_by_indices(dataset=input_data_x, filter_ids=non_zeros)
    add_new_x = filter_tensor_dataset_by_indices(dataset=add_input_data_x, filter_ids=non_zeros)

    if rule_matches_z is not None:
        return new_x, add_new_x, class_probas_y[non_zeros], rule_matches_z[non_zeros]

    return new_x, add_new_x, class_probas_y[non_zeros]


def filter_probability_threshold(
        input_data_x: TensorDataset, class_probas_y: np.ndarray, rule_matches_z: np.ndarray = None,
        probability_threshold: float = 0.7
) -> Union[Tuple[TensorDataset, np.ndarray, np.ndarray], Tuple[TensorDataset, np.ndarray]]:
    """Filters instances where no single class probability exceeds "probability_threshold".
    """
    prob_sums = class_probas_y.max(axis=-1)
    conclusive_idx = np.where(prob_sums >= probability_threshold)[0]

    new_x = filter_tensor_dataset_by_indices(dataset=input_data_x, filter_ids=conclusive_idx)

    if rule_matches_z is not None:
        return new_x, class_probas_y[conclusive_idx], rule_matches_z[conclusive_idx]

    return new_x, class_probas_y[conclusive_idx]


def filter_probability_threshold_two_datasets(
        input_data_x: TensorDataset, add_input_data_x: TensorDataset, class_probas_y: np.ndarray,
        rule_matches_z: np.ndarray = None, probability_threshold: float = 0.7
) -> Union[
    Tuple[TensorDataset, TensorDataset, np.ndarray, np.ndarray], Tuple[TensorDataset, TensorDataset, np.ndarray]]:
    """Filters instances where no single class probability exceeds "probability_threshold".
    """
    prob_sums = class_probas_y.max(axis=-1)
    conclusive_idx = np.where(prob_sums >= probability_threshold)[0]

    new_x = filter_tensor_dataset_by_indices(dataset=input_data_x, filter_ids=conclusive_idx)
    add_new_x = filter_tensor_dataset_by_indices(dataset=add_input_data_x, filter_ids=conclusive_idx)

    if rule_matches_z is not None:
        return new_x, add_new_x, class_probas_y[conclusive_idx], rule_matches_z[conclusive_idx]

    return new_x, add_new_x, class_probas_y[conclusive_idx]


def get_pruned_input(
        rule_matches_z: np.ndarray, model_input_x: np.ndarray, x_mask, noisy_labels
) -> Tuple[Union[np.ndarray, TensorDataset], np.ndarray, np.ndarray]:

    noisy_labels_pruned = noisy_labels[x_mask]
    rule_matches_z_pruned = rule_matches_z[x_mask]

    if isinstance(model_input_x, np.ndarray):
        return model_input_x[x_mask], noisy_labels_pruned, rule_matches_z_pruned

    elif isinstance(model_input_x, TensorDataset):
        # todo: write tests
        model_input_x_pruned_subset = [data for data in Subset(model_input_x, np.where(x_mask)[0])]
        model_input_x_pruned = TensorDataset(*[
            torch.stack(([tensor[i] for tensor in model_input_x_pruned_subset]))
            for i in range(len(model_input_x.tensors))
        ])
        return model_input_x_pruned, noisy_labels_pruned, rule_matches_z_pruned

    else:
        raise ValueError("Unknown input format")
