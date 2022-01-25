import numpy as np

from knodle.transformation.filter import filter_empty_probabilities, filter_probability_threshold
from tests.transformation.generic import filter_input


def test_filter_empty_probabilities(filter_input):
    input_dataset, class_probs = filter_input

    gold_ids = np.ones((2, 4))
    gold_ids[0, 0] = 0
    gold_mask = np.ones((2, 4))
    gold_mask[1, 1] = 0
    gold_probs = np.array([
        [0.5, 0.5],
        [0.3, 0.7]
    ])

    new_input_dataset, new_probs = filter_empty_probabilities(input_dataset, class_probs)

    assert np.array_equal(new_input_dataset.tensors[0].detach().numpy(), gold_ids)
    assert np.array_equal(new_input_dataset.tensors[1].detach().numpy(), gold_mask)
    assert np.array_equal(new_probs, gold_probs)


def test_filter_probability_threshold(filter_input):
    input_dataset, class_probs = filter_input

    gold_ids = np.ones((1, 4))
    gold_mask = np.ones((1, 4))
    gold_mask[0, 1] = 0
    gold_probs = np.array([
        [0.3, 0.7]
    ])

    new_input_dataset, new_probs = filter_probability_threshold(input_dataset, class_probs, probability_threshold=0.6)

    assert np.array_equal(new_input_dataset.tensors[0].detach().numpy(), gold_ids)
    assert np.array_equal(new_input_dataset.tensors[1].detach().numpy(), gold_mask)
    assert np.array_equal(new_probs, gold_probs)

