import pytest

import numpy as np
import torch
from torch.utils.data import TensorDataset

from knodle.model.logistic_regression_model import LogisticRegressionModel


@pytest.fixture
def std_trainer_input_1():
    num_samples = 64
    num_features = 16
    num_rules = 6
    num_classes = 2

    x_np = np.ones((num_samples, num_features)).astype(np.float32)
    x_tensor = torch.from_numpy(x_np)
    model_input_x = TensorDataset(x_tensor)

    rule_matches_z = np.zeros((num_samples, num_rules))
    rule_matches_z[0, 0] = 1
    rule_matches_z[1:, 1] = 1

    mapping_rules_labels_t = np.zeros((num_rules, num_classes))
    mapping_rules_labels_t[:, 1] = 1

    y_np = np.ones((num_samples,))
    y_labels = TensorDataset(torch.from_numpy(y_np))

    model = LogisticRegressionModel(num_features, num_classes)

    return (
        model,
        model_input_x, rule_matches_z, mapping_rules_labels_t,
        y_labels
    )


@pytest.fixture
def std_trainer_input_2():
    model = LogisticRegressionModel(5, 2)

    inputs_x = TensorDataset(torch.Tensor(np.array([[1, 1, 1, 1, 1],
                                                    [2, 2, 2, 2, 2],
                                                    [3, 3, 3, 3, 3],
                                                    [6, 6, 6, 6, 6],
                                                    [7, 7, 7, 7, 7]])))

    mapping_rules_labels_t = np.array([[1, 0], [1, 0], [0, 1]])
    train_rule_matches_z = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1]])

    test_dataset = TensorDataset(torch.Tensor(np.array([[4, 4, 4, 4, 4], [5, 5, 5, 5, 5]])))
    test_labels = TensorDataset(torch.Tensor(np.array([0, 1])))

    return (
        model,
        inputs_x, mapping_rules_labels_t, train_rule_matches_z,
        test_dataset, test_labels
    )
