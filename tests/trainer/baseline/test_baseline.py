import numpy as np
import torch

from knodle.model.logistic_regression.logistic_regression_model import (
    LogisticRegressionModel,
)
from knodle.trainer.baseline.baseline import SimpleDsModelTrainer


def test_train():

    num_samples = 64
    num_features = 16
    num_rules = 6
    num_classes = 2

    model = LogisticRegressionModel(num_features, num_classes)

    x_np = np.ones((num_samples, num_features)).astype(np.float32)
    x_tensor = torch.from_numpy(x_np)
    x = torch.utils.data.TensorDataset(x_tensor)

    z = np.zeros((num_samples, num_rules))
    z[0, 0] = 1
    z[1:, 1] = 1

    t = np.zeros((num_rules, num_classes))
    t[:, 0] = 1

    trainer = SimpleDsModelTrainer(
        model=model, mapping_rules_labels_t=t, model_input_x=x, rule_matches_z=z
    )

    trainer.train()

    y_np = np.ones((num_samples, 1))
    y_tensor = torch.from_numpy(y_np)
    a = trainer.test(x, torch.utils.data.TensorDataset(y_tensor))

    # We train 100% on 1 class, thus test accuracy should be 100%
    assert a["accuracy"] == 1
