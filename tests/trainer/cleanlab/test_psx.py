import numpy as np
# todo: test compute psx matrix -> check also that there is no duplicates
#  psx[indices_holdout_cv] = psx_cv : value for each sample is filled only once

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset

from knodle.trainer.ulf.noisy_matrix_estimation import calibrate_confident_joint_rule2class
from knodle.trainer.cleanlab.psx_estimation import compute_psx_matrix

from knodle.trainer.cleanlab.config import CleanLabConfig
from knodle.trainer.cleanlab.utils import calculate_threshold


def test_threshold_calculation_without_class():
    psx = np.array([[1., 2.], [5., 6.], [4., 3.]])
    labels = np.array([1., 1., 0.])
    true_threshold = np.array([4., 4.])

    threshold = calculate_threshold(psx, labels)

    assert np.array_equal(threshold, true_threshold)


def test_threshold_calculation_with_class():
    psx = np.array([[1., 2.], [5., 6.], [4., 3.]])
    labels = np.array([1., 1., 0.])
    true_threshold = np.array([4., 4.])
    num_classes = 2
    threshold = calculate_threshold(psx, labels, num_classes)
    assert np.array_equal(threshold, true_threshold)


def test_compute_psx_matrix(std_trainer_input_2):
    model, inputs_x, mapping_rules_labels_t, train_rule_matches_z, test_dataset, test_labels = std_trainer_input_2

    config = CleanLabConfig(batch_size=32, seed=123, output_classes=2,  criterion=CrossEntropyLoss)

    cv_train_dataset_1 = TensorDataset(torch.IntTensor(np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]])),
                                       torch.LongTensor(np.array([0, 0])))      # placeholder

    cv_holdout_datasets_1 = TensorDataset(torch.IntTensor(np.array([[3, 3, 3, 3, 3, 3]])),
                                          torch.IntTensor(np.array([2])),
                                          torch.LongTensor(np.array([0])))      # placeholder

    cv_train_dataset_2 = TensorDataset(torch.IntTensor(np.array([[3, 3, 3, 3, 3, 3]])),
                                       torch.LongTensor(np.array([0])))         # placeholder

    cv_holdout_datasets_2 = TensorDataset(torch.Tensor(np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]])),
                                          torch.IntTensor(np.array([0, 1])),
                                          torch.LongTensor(np.array([0, 0])))       # placeholder

    cv_train_datasets = [cv_train_dataset_1, cv_train_dataset_2]
    cv_holdout_datasets = [cv_holdout_datasets_1, cv_holdout_datasets_2]

    psx = compute_psx_matrix(model, cv_train_datasets, cv_holdout_datasets, num_samples=3, config=config)

    psx_gold_shape = (3, 2)

    assert psx.shape == psx_gold_shape


def test_calibrate_confident_joint_rule2class():
    confident_joint = np.array([[2, 1], [1, 1], [0, 1]])
    rule_matches_z = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1]])

    calibrated_cj = calibrate_confident_joint_rule2class(confident_joint, rule_matches_z)
    true_calibrated_cj = np.array([[2, 1], [1, 1], [0, 2]])

    assert np.array_equal(calibrated_cj, true_calibrated_cj)


