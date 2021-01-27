import pytest
import numpy as np

from knodle.trainer.utils.denoise import get_majority_vote_probs, get_majority_vote_labels


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
    assert np.array_equal(majority_labels,  gold_labels)


def test_get_majority_vote_probs(values):
    z, t, gold_probs, _ = values

    majority_probs = get_majority_vote_probs(z, t)
    assert np.array_equal(gold_probs, majority_probs)
