import numpy as np
from numpy.testing import assert_array_equal

from knodle.trainer.utils.denoise import get_majority_vote_probs, activate_all_neighbors


def test_majority_vote():
    rule_matches_z = np.array([[1, 1], [1, 1], [0, 0]])
    mapping_rules_labels_t = np.array([[0], [1]])
    majority_vote = get_majority_vote_probs(rule_matches_z, mapping_rules_labels_t)
    correct_result = np.array([[1], [1], [0]])
    assert_array_equal(correct_result, majority_vote)


def test_denoise_knn():
    test_array = np.array([[0, 1], [1, 0]])
    right_result = np.array([[1, 1], [1, 1]])
    indices = np.array([[0, 1], [1, 0]])
    denoised_z = activate_all_neighbors(test_array, indices)
    assert_array_equal(denoised_z, right_result)
