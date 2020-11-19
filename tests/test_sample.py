# Sample Test passing with nose and pytest
import pytest
import numpy as np
from numpy.testing import assert_array_equal


@pytest.fixture
def init_majority_vote_function():
    from knodle.final_label_decider.FinalLabelDecider import (
        get_majority_vote_probabilities,
    )

    return get_majority_vote_probabilities


def test_pass(init_majority_vote_function):
    mock_data = np.array(((1, 1, 1), (1, 1, 0), (1, 0, 0)))
    output_classes = 2
    correct_result = np.array(((0, 1), (0, 1), (1, 0)))
    result = init_majority_vote_function(mock_data, output_classes)
    assert_array_equal(correct_result, result)
