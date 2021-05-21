import numpy as np
import pytest

from cleanlab.latent_estimation import compute_confident_joint
from cleanlab.pruning import get_noise_indices, keep_at_least_n_per_class


@pytest.fixture(scope='session')
def test_data():
    noisy_labels = np.array([0, 0, 1, 0, 1, 1])
    psx = np.array([[0.3, 0.7], [0.8, 0.2], [0.6, 0.4], [0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
    gold_confident_joint_without_calibration = np.array([[2, 1], [1, 1]])
    gold_confident_joint_with_calibration = np.array([[2, 1], [1, 2]])

    gold_label_errors_mask = np.array([])
    gold_prune_count_matrix = np.array([[2, 1], [1, 2]])

    return noisy_labels, psx, gold_confident_joint_without_calibration, gold_confident_joint_with_calibration, \
           gold_label_errors_mask, gold_prune_count_matrix


def test_compute_confident_joint_without_calibration(test_data):
    confident_joint = compute_confident_joint(test_data[0], test_data[1], calibrate=False)
    np.testing.assert_array_equal(confident_joint, test_data[2])


def test_get_noise_indices(test_data):
    label_errors_mask = get_noise_indices(test_data[0], test_data[1], confident_joint=test_data[3])
    np.testing.assert_array_equal(label_errors_mask, test_data[4])


def test_prune_count_matrix(test_data):
    prune_count_matrix = keep_at_least_n_per_class(
        prune_count_matrix=test_data[3].T,
        n=0
    )
    np.testing.assert_array_equal(prune_count_matrix, test_data[5])
