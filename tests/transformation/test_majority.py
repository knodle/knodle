import pytest

import numpy as np

from knodle.transformation.majority import (
    probabilies_to_majority_vote, z_t_matrices_to_majority_vote_probs, z_t_matrices_to_majority_vote_labels
)
from tests.transformation.generic import majority_input


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
        result = probabilies_to_majority_vote(probs, **settings)

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
        result = probabilies_to_majority_vote(probs, **settings)

        assert isinstance(result, int)
        assert result in gold_label


def test_probabilies_to_majority_vote_errors():
    probs, gold_label, settings = (
        np.array([0.0, 0.0, 0.0]), [], {"choose_random_label": False, "other_class_id": None}
    )

    with pytest.raises(ValueError):
        result = probabilies_to_majority_vote(probs, **settings)


def test_get_majority_vote_labels(majority_input):
    z, t, _, gold_labels = majority_input

    majority_labels = z_t_matrices_to_majority_vote_labels(z, t, choose_random_label=False, other_class_id=-1)
    print(majority_labels)
    assert np.array_equal(majority_labels, gold_labels)


def test_get_majority_vote_probs(majority_input):
    z, t, gold_probs, _ = majority_input

    majority_probs = z_t_matrices_to_majority_vote_probs(z, t)
    assert np.array_equal(gold_probs, majority_probs)