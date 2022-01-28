import pytest
import torch
from torch.utils.data import TensorDataset

import numpy as np


@pytest.fixture
def filter_input():
    input_ids = np.ones((3, 4))
    input_ids[0, 0] = 0
    input_mask = np.ones((3, 4))
    input_mask[1, 1] = 0
    class_probs = np.array([
        [0.5, 0.5],
        [0.3, 0.7],
        [0.0, 0.0]
    ])

    input_dataset = TensorDataset(torch.from_numpy(input_ids), torch.from_numpy(input_mask))

    return input_dataset, class_probs


@pytest.fixture
def majority_input():
    z = np.zeros((4, 4))
    t = np.zeros((4, 2))

    z[0, 0] = 1
    z[0, 2] = 1
    z[1, 0] = 1
    z[1, 1] = 1
    z[1, 2] = 1
    z[1, 3] = 1
    z[2, 1] = 1

    t[0, 0] = 1
    t[1, 1] = 1
    t[2, 1] = 1
    t[3, 1] = 1

    gold_probs = np.array([
        [0.5, 0.5],
        [0.25, 0.75],
        [0, 1],
        [0, 0]
    ])

    gold_labels = np.array([-1, 1, 1, -1])

    return z, t, gold_probs, gold_labels