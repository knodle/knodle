import torch
from torch.utils.data import TensorDataset
import pytest
import numpy as np

from knodle.trainer.wscrossweigh.data_splitting_by_rules import get_rules_sample_ids, get_samples_labels_idx_by_rule_id


@pytest.fixture(scope='session')
def get_test_data():

    rule_assignments_t = np.array([[1, 0, 0],
                                   [1, 0, 0],
                                   [0, 1, 0],
                                   [0, 1, 0],
                                   [0, 0, 1],
                                   [0, 0, 1],
                                   [0, 0, 1]])

    inputs_x = TensorDataset(torch.Tensor(np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                                    [1, 1, 1, 1, 1, 1, 1, 1],
                                                    [2, 2, 2, 2, 2, 2, 2, 2],
                                                    [3, 3, 3, 3, 3, 3, 3, 3],
                                                    [4, 4, 4, 4, 4, 4, 4, 4],
                                                    [5, 5, 5, 5, 5, 5, 5, 5],
                                                    [6, 6, 6, 6, 6, 6, 6, 6],
                                                    [7, 7, 7, 7, 7, 7, 7, 7]])))

    rule_matches_z = np.array([[0, 1, 1, 0, 0, 0, 0],
                               [1, 0, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 1, 0],
                               [1, 0, 1, 0, 0, 0, 0],
                               [1, 1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0]])

    rule2sample_id = {0: {1, 5, 6},
                              1: {0, 6},
                              2: {0, 1, 5, 6},
                              3: {1, 6, 7},
                              4: {2},  # no rel sample
                              5: {4},  # no rel sample
                              6: {3}}  # no rel sample

    return inputs_x, rule_matches_z, rule_assignments_t, rule2sample_id


@pytest.fixture(scope='session')
def get_cw_data_test(get_test_data):
    # no filtering, no no_match samples
    test_rules_idx = [1, 2]
    labels = np.array([[0.5, 0.5, 0], [0.3, 0.7, 0], [0, 0, 1], [0, 0, 1],
                       [0, 0, 1], [0.5, 0.5, 0], [0.5, 0.5, 0], [0, 1, 0]])

    return [[get_test_data[0],
             labels,
             test_rules_idx,
             get_test_data[3],
             torch.Tensor(np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1, 1],
                                    [5, 5, 5, 5, 5, 5, 5, 5],
                                    [6, 6, 6, 6, 6, 6, 6, 6]])),
             np.array([[0.5, 0.5, 0], [0.3, 0.7, 0], [0.5, 0.5, 0], [0.5, 0.5, 0]]),
             np.array([0, 1, 5, 6])
             ]]


@pytest.fixture(scope='session')
def get_cw_data_train(get_test_data):
    # rules to be filtered: 1, 2, no_match samples should be added

    test_samples_idx = [0, 6, 1, 5]
    train_rules_idx = [0, 3, 4, 5, 6]  # 1, 5, 6, 7 --> delete inters --> 7 --> + no rel --> 2, 3, 4, 7
    labels = np.array([[0.5, 0.5, 0], [0.3, 0.7, 0], [0, 0, 1], [0, 0, 1],
                       [0, 0, 1], [0.5, 0.5, 0], [0.5, 0.5, 0], [0, 1, 0]])

    return [[get_test_data[0],
             labels,
             train_rules_idx,
             get_test_data[3],
             test_samples_idx,
             torch.Tensor(np.array([[2, 2, 2, 2, 2, 2, 2, 2],
                                    [3, 3, 3, 3, 3, 3, 3, 3],
                                    [4, 4, 4, 4, 4, 4, 4, 4],
                                    [7, 7, 7, 7, 7, 7, 7, 7]])),
             np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0]]),
             np.array([2, 3, 4, 7])
             ]]


def test_sample_ids_matched_rules_correspondence(get_test_data):
    assert get_rules_sample_ids(get_test_data[1]) == get_test_data[3]


def test_get_cw_data_test(get_cw_data_test):
    for data in get_cw_data_test:
        samples, labels, ids = get_samples_labels_idx_by_rule_id(
            data[0], data[1], data[2], data[3]
        )

        assert torch.equal(samples.tensors[0], data[4])
        np.testing.assert_array_equal(labels, data[5])
        np.testing.assert_array_equal(ids, data[6])


def test_get_cw_data_train(get_cw_data_train):
    for data in get_cw_data_train:
        samples, labels, ids = get_samples_labels_idx_by_rule_id(
            data[0], data[1], data[2], data[3], data[4]
        )

        assert torch.equal(samples.tensors[0], data[5])
        np.testing.assert_array_equal(labels, data[6])
        np.testing.assert_array_equal(ids, data[7])


def test_random_check(get_cw_data_train):
    for data in get_cw_data_train:
        samples, labels, ids = get_samples_labels_idx_by_rule_id(
            data[0], data[1], data[2], data[3], data[4]
        )
        rnd_tst = np.random.randint(0, samples.tensors[0].shape[0])  # take some random index
        tst_sample = samples.tensors[0][rnd_tst, :]
        tst_idx = ids[rnd_tst]
        tst_label = labels[rnd_tst, :] if len(labels.shape) > 1 else labels[rnd_tst]

        tst_sample_true = data[5][rnd_tst, :]
        tst_label_true = data[6][rnd_tst, :] if len(data[6].shape) > 1 else data[6][rnd_tst]
        tst_idx_true = data[7][rnd_tst]

        assert torch.equal(tst_sample, tst_sample_true)
        np.testing.assert_array_equal(tst_label, tst_label_true)
        np.testing.assert_array_equal(tst_idx, tst_idx_true)
