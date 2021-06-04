import numpy as np
import pytest

from cleanlab.latent_estimation import compute_confident_joint, estimate_latent
from cleanlab.pruning import get_noise_indices
from cleanlab.util import round_preserving_sum

from knodle.trainer.cleanlab.utils import calculate_sample_weights


NUM_CLASSES = 2


@pytest.fixture(scope='session')
def test_data():
    data_samples = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    data_samples_large = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    noisy_labels = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 1])
    noisy_labels_large = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0])

    psx = np.array([[0.3, 0.7], [0.8, 0.2], [0.6, 0.4], [0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8],
                    [0.4, 0.6], [0.7, 0.3]])
    psx_large = np.array([
        [0.3, 0.7], [0.8, 0.2], [0.6, 0.4], [0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8], [0.4, 0.6],
        [0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.2, 0.8], [0.9, 0.1]])

    gold_confident_joint_without_calibration = np.array([[3, 2], [2, 2]])
    gold_confident_joint_without_calibration_large = np.array([[5, 2], [2, 4]])

    gold_confident_joint_with_calibration = np.array([[3, 2], [2, 3]])
    gold_confident_joint_with_calibration_large = np.array([[5, 2], [3, 5]])

    gold_labels_pruned = np.array([0, 1, 0, 1, 0, 1])
    gold_labels_pruned_large = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0])

    gold_data_samples_pruned = np.array([1, 2, 3, 5, 6, 7])
    gold_data_samples_pruned_large = np.array([1, 3, 5, 6, 7, 10, 12, 13, 14])

    return [[data_samples, noisy_labels, psx, gold_confident_joint_without_calibration,
            gold_confident_joint_with_calibration, gold_labels_pruned, gold_data_samples_pruned],

            [data_samples_large, noisy_labels_large, psx_large, gold_confident_joint_without_calibration_large,
             gold_confident_joint_with_calibration_large, gold_labels_pruned_large, gold_data_samples_pruned_large]]


def test_compute_confident_joint_without_calibration(test_data):
    for data in test_data:
        confident_joint = compute_confident_joint(data[1], data[2], calibrate=False)
        np.testing.assert_array_equal(confident_joint, data[3])


def test_compute_confident_joint_with_calibration(test_data):
    for data in test_data:
        confident_joint = compute_confident_joint(data[1], data[2], calibrate=True)
        np.testing.assert_array_equal(confident_joint, data[4])


def test_get_noise_indices(test_data):
    for data in test_data:
        label_errors_mask = get_noise_indices(data[1], data[2], confident_joint=data[4], n_jobs=1)
        x_mask = ~label_errors_mask
        labels_pruned = data[1][x_mask]
        data_samples_pruned = data[0][x_mask]

        np.testing.assert_array_equal(labels_pruned, data[5])
        np.testing.assert_array_equal(data_samples_pruned, data[6])


def test_calculate_sample_weights(test_data):
    sample_weights = []
    for data in test_data:
        noisy_labels, psx, confident_joint, noisy_labels_pruned = data[1], data[2], data[4], data[5]
        _, noise_matrix, _ = estimate_latent(confident_joint=confident_joint, s=noisy_labels)
        sample_weight = calculate_sample_weights(NUM_CLASSES, noise_matrix, noisy_labels_pruned)
        sample_weights.append(sample_weight)
    print(sample_weights)


def test_round_preserving_sum():
    # this CL function rounds the float to ints while preserving the original sum.
    # The items with the largest difference are adjusted to preserve the sum.
    float_array = np.array([2.5, 2.1, 4.5, 3.8, 2.5, 1.3])
    true_rounded_array = np.array([2., 2., 5., 4., 3., 1.])

    rounded = round_preserving_sum(float_array)
    np.testing.assert_array_equal(rounded, true_rounded_array)

