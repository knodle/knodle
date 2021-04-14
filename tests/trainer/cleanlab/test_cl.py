import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset

from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.cleanlab.cleanlab import CleanLabTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig


def test_cleanlab_base_test():

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

    config = CleanLabConfig(cv_n_folds=2, criterion=CrossEntropyLoss)

    trainer = CleanLabTrainer(
        model=model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=inputs_x,
        rule_matches_z=train_rule_matches_z,
        trainer_config=config
    )

    trainer.train()
    clf_report, _ = trainer.test(test_dataset, test_labels)

    # Check that this runs without error
    assert True
