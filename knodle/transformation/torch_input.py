import numpy as np
import torch
from torch.utils.data import TensorDataset


def input_labels_to_tensordataset(model_input_x: TensorDataset, labels: np.ndarray) -> TensorDataset:
    """
    This function takes Dataset with data features (num_samples x features dimension x features) and
    labels (num_samples x labels dimension) and turns it into one Dataset
    """
    model_tensors = model_input_x.tensors
    input_label_dataset = TensorDataset(*model_tensors, torch.from_numpy(labels))

    return input_label_dataset


def input_info_labels_to_tensordataset(
        model_input_x: TensorDataset, input_info: np.ndarray, labels: np.ndarray
) -> TensorDataset:
    """
    This function takes Dataset with data features (num_samples x features dimension x features),
    labels (num_samples x labels dimension) and some additional information about data encoded as
    a numpy array (num_samples x n; could be sample weights, sample indices etc) and turns it into one Dataset
    """
    model_tensors = model_input_x.tensors
    input_ids_label_dataset = TensorDataset(*model_tensors, torch.from_numpy(input_info), torch.from_numpy(labels))

    return input_ids_label_dataset
