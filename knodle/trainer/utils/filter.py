import numpy as np
import torch
from torch.utils.data import TensorDataset


def filter_empty_probabilities(input_data_x: TensorDataset, class_probas_y: TensorDataset):
    """Delete rows of TensorDataset's where the cumulative probability equals 0.

    :param input_data_x: A TensorDataset serving as input to a model
    :param class_probas_y: A TensorDataset serving as gold standard
    :return: Modified TensorDataset's
    """
    y_arr = class_probas_y.tensors[0].detach().numpy()

    if len(y_arr.shape) != 2:
        raise ValueError("y_probs needs to be a matrix of dimensions num_samples x num_classes")

    prob_sums = y_arr.sum(axis=-1)
    non_zeros = np.where(prob_sums != 0)[0]

    new_tensors = []
    for i in range(len(input_data_x.tensors)):
        new_tensors.append(input_data_x.tensors[i][non_zeros])

    new_x = TensorDataset(*new_tensors)
    new_y = TensorDataset(torch.from_numpy(y_arr[non_zeros]))

    return new_x, new_y

