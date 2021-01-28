import numpy as np
from numpy.testing import assert_array_equal


def test_majority_vote():
    rule_matches_z = np.array([[1, 1], [1, 1], [0, 0]])
    mapping_rules_labels_t = np.array([[0], [1]])
    majority_vote = get_majority_vote_probs(rule_matches_z, mapping_rules_labels_t)
    correct_result = np.array([[1], [1], [0]])
    assert_array_equal(correct_result, majority_vote)
