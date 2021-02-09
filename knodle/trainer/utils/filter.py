import numpy as np
from torch.utils.data import TensorDataset


def filter_empty_probabilities(input_data_x: TensorDataset, class_probas_y: np.array):
    """Delete rows of TensorDataset's where the cumulative probability equals 0.

    Args:
        input_data_x: A TensorDataset serving as input to a model
        class_probas_y: Array, holding class probabilities, shape=num_samples, num_classes
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
    return new_x, class_probas_y[non_zeros]


def filter_empty_probabilities_x_y_z(input_data_x: TensorDataset, class_probas_y: np.array, rule_matches_z: np.array):
    """Delete rows of TensorDataset's where the cumulative probability equals 0.

    Args:
        input_data_x: A TensorDataset serving as input to a model
        class_probas_y: Array, holding class probabilities, shape=num_samples, num_classes
    :return: Modified TensorDataset's
    """

    if len(class_probas_y.shape) != 2:
        raise ValueError("y_probs needs to be a matrix of dimensions num_samples x num_classes")

    prob_sums = class_probas_y.sum(axis=-1)
    non_zeros = np.where(prob_sums != 0)[0]

    new_tensors, new_z = [], []
    for i in range(len(input_data_x.tensors)):
        new_tensors.append(input_data_x.tensors[i][non_zeros])

    new_x = TensorDataset(*new_tensors)
    return new_x, rule_matches_z[non_zeros], class_probas_y[non_zeros]
