import pytest
import numpy as np
import torch
from torch.utils.data import TensorDataset

from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.wscrossweigh.wscrossweigh_weights_calculator import WSCrossWeighWeightsCalculator


@pytest.fixture(scope='session')
def get_test_data():
    model = LogisticRegressionModel(8, 3)
    rule_assignments_t = np.array([[1, 0, 0],
                                   [1, 0, 0],
                                   [0, 1, 0],
                                   [0, 1, 0],
                                   [0, 0, 1],
                                   [0, 0, 1],
                                   [0, 0, 1]])

    inputs_x = TensorDataset(torch.Tensor(np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                                                    [2, 2, 2, 2, 2, 2, 2, 2],
                                                    [3, 3, 3, 3, 3, 3, 3, 3],
                                                    [4, 4, 4, 4, 4, 4, 4, 4],
                                                    [5, 5, 5, 5, 5, 5, 5, 5],
                                                    [6, 6, 6, 6, 6, 6, 6, 6],
                                                    [7, 7, 7, 7, 7, 7, 7, 7],
                                                    [8, 8, 8, 8, 8, 8, 8, 8]])))

    rule_matches_z = np.array([[0, 1, 1, 0, 0, 0, 0],
                               [1, 0, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 1, 0],
                               [1, 0, 1, 0, 0, 0, 0],
                               [1, 1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0]])

    rules_samples_ids_dict = {0: {1, 5, 6},
                              1: {0, 6},
                              2: {0, 1, 5, 6},
                              3: {1, 6, 7},
                              4: {2},  # no rel sample
                              5: {4},  # no rel sample
                              6: {3}}  # no rel sample

    trainer = WSCrossWeighWeightsCalculator(
        model=model,
        mapping_rules_labels_t=rule_assignments_t,
        model_input_x=inputs_x,
        rule_matches_z=rule_matches_z
    )

    return trainer, rules_samples_ids_dict


@pytest.fixture(scope='session')
def get_cw_data_test(get_test_data):
    # no filtering, no no_match samples
    test_rules_idx = np.array([1, 2])
    labels = np.array([[0.5, 0.5, 0], [0.3, 0.7, 0], [0, 0, 1], [0, 0, 1],
                       [0, 0, 1], [0.5, 0.5, 0], [0.5, 0.5, 0], [0, 1, 0]])

    return [[get_test_data[0],
             labels,
             test_rules_idx,
             get_test_data[1],
             torch.Tensor(np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                                    [2, 2, 2, 2, 2, 2, 2, 2],
                                    [6, 6, 6, 6, 6, 6, 6, 6],
                                    [7, 7, 7, 7, 7, 7, 7, 7]])),
             np.array([[0.5, 0.5, 0], [0.3, 0.7, 0], [0.5, 0.5, 0], [0.5, 0.5, 0]]),
             np.array([0, 1, 5, 6])
             ]]


@pytest.fixture(scope='session')
def get_cw_data_train(get_test_data):
    # rules to be filtered: 1, 2, no_match samples should be added

    test_samples_idx = np.array([0, 6, 1, 5])
    train_rules_idx = np.array([0, 3, 4, 5, 6])  # 1, 5, 6, 7 --> delete inters --> 7 --> + no rel --> 2, 3, 4, 7
    labels = np.array([[0.5, 0.5, 0], [0.3, 0.7, 0], [0, 0, 1], [0, 0, 1],
                       [0, 0, 1], [0.5, 0.5, 0], [0.5, 0.5, 0], [0, 1, 0]])

    return [[get_test_data[0],
             labels,
             train_rules_idx,
             get_test_data[1],
             test_samples_idx,
             torch.Tensor(np.array([[3, 3, 3, 3, 3, 3, 3, 3],
                                    [4, 4, 4, 4, 4, 4, 4, 4],
                                    [5, 5, 5, 5, 5, 5, 5, 5],
                                    [8, 8, 8, 8, 8, 8, 8, 8]])),
             np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0]]),
             np.array([2, 3, 4, 7])
             ]]


def test_sample_ids_matched_rules_correspondence(get_test_data):
    # for data in get_sample_ids_matched_rules_correspondence:
    #     assert WSCrossWeighWeightsCalculator._get_rules_samples_ids_dict(data[0]) == data[1]
    assert WSCrossWeighWeightsCalculator._get_rules_samples_ids_dict(get_test_data[0]) == get_test_data[1]


def test_get_cw_data_test(get_cw_data_test):
    for data in get_cw_data_test:
        samples, labels, ids = WSCrossWeighWeightsCalculator._get_cw_samples_labels_idx(
            data[0], data[1], data[2], data[3]
        )

        assert torch.equal(samples.tensors[0], data[4])
        np.testing.assert_array_equal(labels, data[5])
        np.testing.assert_array_equal(ids, data[6])


def test_get_cw_data_train(get_cw_data_train):
    for data in get_cw_data_train:
        samples, labels, ids = WSCrossWeighWeightsCalculator._get_cw_samples_labels_idx(
            data[0], data[1], data[2], data[3], data[4]
        )

        assert torch.equal(samples.tensors[0], data[5])
        np.testing.assert_array_equal(labels, data[6])
        np.testing.assert_array_equal(ids, data[7])
