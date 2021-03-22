import numpy as np
import torch
from torch.utils.data import TensorDataset

from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.baseline.majority import MajorityVoteTrainer


def test_train():
    num_samples = 64
    num_features = 16
    num_rules = 6
    num_classes = 2

    model = LogisticRegressionModel(num_features, num_classes)

    x_np = np.ones((num_samples, num_features)).astype(np.float32)
    x_tensor = torch.from_numpy(x_np)
    model_input_x = TensorDataset(x_tensor)

    rule_matches_z = np.zeros((num_samples, num_rules))
    rule_matches_z[0, 0] = 1
    rule_matches_z[1:, 1] = 1

    mapping_rules_labels_t = np.zeros((num_rules, num_classes))
    mapping_rules_labels_t[:, 1] = 1

    trainer = MajorityVoteTrainer(
        model=model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=model_input_x,
        rule_matches_z=rule_matches_z,
    )

    trainer.train()

    y_np = np.ones((num_samples,))
    y_labels = TensorDataset(torch.from_numpy(y_np))
    metrics, _ = trainer.test(model_input_x, y_labels)

    # Check whether the code ran up to here
    assert 2 == 2
