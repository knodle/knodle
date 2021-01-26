import numpy as np
import torch
from torch.utils.data import TensorDataset

from knodle.trainer.utils.filter import filter_empty_probabilities


def test_filter_empty_probabilities():
    input_ids = np.ones((3, 4))
    input_ids[0, 0] = 0
    input_mask = np.ones((3, 4))
    input_mask[1, 1] = 0
    class_probs = np.array([
        [0.5, 0.5],
        [0.3, 0.7],
        [0.0, 0.0]
    ])

    gold_ids = np.ones((2, 4))
    gold_ids[0, 0] = 0
    gold_mask = np.ones((2, 4))
    gold_mask[1, 1] = 0
    gold_probs = np.array([
        [0.5, 0.5],
        [0.3, 0.7]
    ])

    input_dataset = TensorDataset(torch.from_numpy(input_ids), torch.from_numpy(input_mask))
    probs_dataset = TensorDataset(torch.from_numpy(class_probs))

    new_input_dataset, new_probs_dataset = filter_empty_probabilities(input_dataset, probs_dataset)

    assert np.array_equal(new_input_dataset.tensors[0].detach().numpy(), gold_ids)
    assert np.array_equal(new_input_dataset.tensors[1].detach().numpy(), gold_mask)
    assert np.array_equal(new_probs_dataset.tensors[0].detach().numpy(), gold_probs)


