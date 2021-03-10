import pytest
import numpy as np
from numpy.testing import assert_array_equal

from knodle.trainer.utils.denoise import activate_neighbors


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

def test_match_persistance():
    test_array = np.array([[0, 1], [1, 0]])
    right_result = np.array([[1, 1], [1, 1]])
    indices = np.array([[0, 1], [1, 0]])

    denoised_z = activate_neighbors(test_array, indices)

    # make sure all old matches persist and have same match values (currently always 1)
    assert denoised_z[test_array == 1].sum() == test_array[test_array == 1].sum()

