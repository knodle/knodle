import random

import pytest

import numpy as np

from knodle.transformation._majority_depr import (
    probabilities_to_majority_vote, z_t_matrices_to_majority_vote_probs, z_t_matrices_to_majority_vote_labels,
    probabilities_to_binary_multi_labels
)


def test_probabilies_to_majority_vote_fixed():
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


def test_probabilies_to_majority_vote_errors():
    probs, gold_label, settings = (
        np.array([0.0, 0.0, 0.0]), [], {"choose_random_label": False, "other_class_id": None}
    )

    with pytest.raises(ValueError):
        result = probabilities_to_majority_vote(probs, **settings)


@pytest.fixture
def prob_values():
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


def test_get_majority_vote_labels(prob_values):
    z, t, _, gold_labels = prob_values

    majority_labels = z_t_matrices_to_majority_vote_labels(z, t, choose_random_label=False, other_class_id=-1)
    print(majority_labels)
    assert np.array_equal(majority_labels, gold_labels)


def test_get_majority_vote_probs(prob_values):
    z, t, gold_probs, _ = prob_values

    majority_probs = z_t_matrices_to_majority_vote_probs(z, t)
    assert np.array_equal(gold_probs, majority_probs)


def test_get_majority_vote_probs_sigmoid(prob_values):
    z, t, _, _ = prob_values

    gold_probs = np.array([
        [[0.73, 0.73], [0.73, 0.95], [0, 0.73], [0, 0]]
    ])

    majority_probs = z_t_matrices_to_majority_vote_probs(z, t, normalization="sigmoid")
    assert np.allclose(gold_probs, majority_probs, rtol=1e-02)


def test_get_majority_vote_probs_error(prob_values):
    z, t, gold_probs, _ = prob_values
    with pytest.raises(
            ValueError,
            match="Unknown label probabilities normalization; currently softmax and sigmoid normalization are supported"
    ):
        _ = z_t_matrices_to_majority_vote_probs(z, t, normalization="relu")


def test_probabilities_to_binary_multi_labels():

    probs1 = np.array([[0.6, 0.8, 0]])
    gold_probs1 = np.array([[1, 1, 0]])

    probs2 = np.array([[0.5, 0.9, 0.8]])
    gold_probs2 = np.array([[0, 1, 0]])

    probs1 = probabilities_to_binary_multi_labels(probs1)
    assert np.array_equal(probs1, gold_probs1)

    probs2 = probabilities_to_binary_multi_labels(probs2, threshold=0.9)
    assert np.array_equal(probs2, gold_probs2)


def test_probabilities_to_binary_multi_labels_random():
    random.seed(10)
    probs = np.array([[0, 0, 0]])
    gold_probs = np.array([[0, 0, 1]])
    probs3 = probabilities_to_binary_multi_labels(probs, choose_random_label=True)
    assert np.array_equal(probs3, gold_probs)


def test_probabilities_to_binary_multi_labels_other():

    probs = np.array([[0, 0, 0]])
    gold_probs = np.array([[1, 0, 0]])

    probs3 = probabilities_to_binary_multi_labels(probs, choose_random_label=False, other_class_id=0)
    assert np.array_equal(probs3, gold_probs)
