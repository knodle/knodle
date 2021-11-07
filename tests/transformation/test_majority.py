import pytest

import numpy as np
import torch
from torch.utils.data import TensorDataset

from knodle.transformation.majority import (
    probabilities_to_majority_vote, z_t_matrices_to_majority_vote_probs, input_to_majority_vote_input
)


def test_probabilies_to_majority_vote_errors():
    with pytest.raises(ValueError) as execinfo_1:
        input_to_majority_vote_input(np.array([0.0]), np.array([0.0]), model_input_x=None, filter_non_labelled=True)
    assert str(execinfo_1.value) == "In order to filter non labeled samples, please provide X matrix."

    with pytest.raises(ValueError) as execinfo_2:
        input_to_majority_vote_input(
            np.array([0.0]), np.array([0.0]), model_input_x=None, probability_threshold=True, filter_non_labelled=False
        )
    assert str(execinfo_2.value) == "In order to filter non labeled samples, please provide X matrix."

    with pytest.raises(ValueError) as execinfo_3:
        input_to_majority_vote_input(
            np.array([0.0]), np.array([0.0]), filter_non_labelled=True, probability_threshold=True
        )
    assert str(execinfo_3.value) == \
           "You can either filter all non labeled samples or those with probabilities below threshold."

    with pytest.raises(ValueError) as execinfo_4:
        input_to_majority_vote_input(np.array([0.0]), np.array([0.0]), other_class_id=0, filter_non_labelled=True)
    assert str(execinfo_4.value) == "You can either filter samples with no weak labels or add them to the other class."


def test_probabilities_to_majority_vote_fixed():
    # format: (probabilities, gold_result, settings)
    probs_gold_result_settings = [
        (np.array([0.5, 0.2, 0.3]), 0, {"choose_random_label": False, "other_class_id": -1}),
        (np.array([0.5, 0.2, 0.3]), 0, {"choose_random_label": True, "other_class_id": None}),
        (np.array([0.5, 0.2, 0.3]), 0, {"choose_random_label": False, "other_class_id": -1}),
        (np.array([0.5, 0.2, 0.3]), 0, {"choose_random_label": False, "other_class_id": None}),

        (np.array([0.5, 0.5, 0.0]), -1, {"choose_random_label": None, "other_class_id": -1}),
        (np.array([0.0, 0.0, 0.0]), -1, {"choose_random_label": None, "other_class_id": -1})
    ]

    for probs, gold_label, settings in probs_gold_result_settings:
        result = probabilities_to_majority_vote(probs, **settings)

        assert isinstance(result, int)
        assert result == gold_label


def test_probabilies_to_majority_vote_random():
    # format: (probabilities, gold_result, settings)
    probs_gold_result_settings = [
        (np.array([0.5, 0.5, 0.0]), [0, 1], {"choose_random_label": True, "other_class_id": None}),

        (np.array([0.3, 0.3, 0, 2, 0.3]), [0, 1, 3], {"choose_random_label": True, "other_class_id": None}),

        (np.array([0.0, 0.0, 0.0]), [-1], {"choose_random_label": False, "other_class_id": -1}),
        (np.array([0.0, 0.0, 0.0]), [0, 1, 2], {"choose_random_label": True, "other_class_id": None})
    ]

    for probs, gold_label, settings in probs_gold_result_settings:
        result = probabilities_to_majority_vote(probs, **settings)

        assert isinstance(result, int)
        assert result in gold_label


@pytest.fixture
def prob_values():
    z = np.zeros((4, 4))
    t = np.zeros((4, 2))

    x = TensorDataset(torch.Tensor(np.random.randint(5, size=(4, 5))))

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

    gold_labels = np.array([2, 1, 1, 2])

    return z, t, x, gold_probs, gold_labels


@pytest.fixture
def prob_values_filtered():

    z = np.zeros((4, 4))
    z_filtered = z[:3, :]

    t = np.zeros((4, 2))

    x = np.random.randint(5, size=(4, 5))
    x_filtered = x[:3, :]
    x = TensorDataset(torch.Tensor(x))
    x_filtered = TensorDataset(torch.Tensor(x_filtered))

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

    gold_probs_filtered = np.array([
        [0.5, 0.5],
        [0.25, 0.75],
        [0, 1]
    ])

    gold_labels_filtered = np.array([2, 1, 1])

    return z, t, x, x_filtered, z_filtered, gold_probs_filtered, gold_labels_filtered


def test_get_majority_vote_probs(prob_values):
    z, t, _, gold_probs, _ = prob_values

    majority_probs = z_t_matrices_to_majority_vote_probs(z, t)
    assert np.array_equal(gold_probs, majority_probs)


def test_input_to_majority_vote_input_labels(prob_values):
    z, t, x, _, gold_labels = prob_values

    model_input_x, noisy_y_train, rule_matches_z = input_to_majority_vote_input(
        z, t, x, other_class_id=2, use_probabilistic_labels=False, filter_non_labelled=False
    )

    assert torch.all(model_input_x.tensors[0].eq(x.tensors[0]))
    assert np.array_equal(gold_labels, noisy_y_train)
    assert np.array_equal(z, rule_matches_z)


def test_input_to_majority_vote_input_probs(prob_values):
    z, t, x, gold_probs, _ = prob_values
    model_input_x, noisy_y_train, rule_matches_z = input_to_majority_vote_input(
        z, t, x, use_probabilistic_labels=True, filter_non_labelled=False
    )

    assert torch.all(model_input_x.tensors[0].eq(x.tensors[0]))
    assert np.array_equal(gold_probs, noisy_y_train)
    assert np.array_equal(z, rule_matches_z)


def test_input_to_majority_vote_input_filter_label(prob_values_filtered):

    z, t, x, x_filtered, z_filtered, _, gold_labels_filtered = prob_values_filtered

    model_input_x, y_labels_filtered, rule_matches_z = input_to_majority_vote_input(
        z, t, x, use_probabilistic_labels=False, filter_non_labelled=True
    )

    # no assertion for label since the ties are broken randomly
    assert torch.all(model_input_x.tensors[0].eq(x_filtered.tensors[0]))
    assert np.array_equal(z_filtered, rule_matches_z)


def test_input_to_majority_vote_input_filter_probs(prob_values_filtered):

    z, t, x, x_filtered, z_filtered, gold_labels_filtered, gold_probs_filtered = prob_values_filtered

    model_input_x, y_labels_filtered, rule_matches_z = input_to_majority_vote_input(
        z, t, x, use_probabilistic_labels=True, filter_non_labelled=True
    )

    assert torch.all(model_input_x.tensors[0].eq(x_filtered.tensors[0]))
    assert np.array_equal(y_labels_filtered, gold_labels_filtered)
    assert np.array_equal(z_filtered, rule_matches_z)
