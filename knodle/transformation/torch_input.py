import numpy as np
import torch
from torch.utils.data import TensorDataset


def input_labels_to_tensordataset(model_input_x: TensorDataset, labels: np.ndarray) -> TensorDataset:

    model_tensors = model_input_x.tensors
    input_label_dataset = TensorDataset(*model_tensors, torch.from_numpy(labels))

    return input_label_dataset


def input_info_labels_to_tensordataset(
        model_input_x: TensorDataset, input_info: np.ndarray, labels: np.ndarray
) -> TensorDataset:

    model_tensors = model_input_x.tensors
    input_ids_label_dataset = TensorDataset(*model_tensors, torch.from_numpy(input_info), torch.from_numpy(labels))

    return input_ids_label_dataset
