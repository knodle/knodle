import pytest
import numpy as np
from numpy.testing import assert_array_equal

from knodle.trainer.utils.denoise import get_majority_vote_probs, get_majority_vote_labels, activate_neighbors


@pytest.fixture
def values():
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


def test_get_majority_vote_labels(values):
    z, t, _, gold_labels = values

    majority_labels = get_majority_vote_labels(z, t)
    assert np.array_equal(majority_labels, gold_labels)


def test_get_majority_vote_probs(values):
    z, t, gold_probs, _ = values

    majority_probs = get_majority_vote_probs(z, t)
    assert np.array_equal(gold_probs, majority_probs)


def test_majority_vote_old():
    rule_matches_z = np.array([[1, 1], [1, 1], [0, 0]])
    mapping_rules_labels_t = np.array([[0], [1]])
    majority_vote = get_majority_vote_probs(rule_matches_z, mapping_rules_labels_t)
    correct_result = np.array([[1], [1], [0]])
    assert_array_equal(correct_result, majority_vote)


def test_denoise_knn():
    test_array = np.array([[0, 1], [1, 0]])
    right_result = np.array([[1, 1], [1, 1]])
    indices = np.array([[0, 1], [1, 0]])

    denoised_z = activate_neighbors(test_array, indices)
    assert_array_equal(denoised_z, right_result)

    test_array = np.diag(np.ones((3,)))
    indices = np.array([
        np.array([0, 1, 2]),
        np.array([0]),
        np.array([1])
    ])

    right_result = np.zeros((3, 3))
    right_result[0, :] = 1
    right_result[1, 0] = 1
    right_result[2, 1] = 1

    denoised_z = activate_neighbors(test_array, indices)
    assert_array_equal(denoised_z, right_result)

    test_array = np.diag(np.ones((3,)))
    indices = np.array([
        np.array([0, 1]),
        np.array([1, 2]),
        np.array([0, 2])
    ])

    right_result = np.ones((3, 3))
    right_result[0, 2] = 0
    right_result[1, 0] = 0
    right_result[2, 1] = 0

    denoised_z = activate_neighbors(test_array, indices)
    assert_array_equal(denoised_z, right_result)
