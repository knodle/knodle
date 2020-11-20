# Sample Test passing with nose and pytest
import pytest


@pytest.fixture
def init_sum_function():
    from knodle.sum import test_sum_function
    return test_sum_function


def test_pass(init_sum_function):
    assert init_sum_function(1,1) == 2