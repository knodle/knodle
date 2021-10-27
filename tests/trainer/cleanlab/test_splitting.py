import random

import pytest

import numpy as np
import torch
from torch.utils.data import TensorDataset

from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.cleanlab.cleanlab import CleanLabTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig
from knodle.trainer.utils.split import k_folds_splitting_by_rules, get_train_test_datasets_by_rule_indices
from knodle.transformation.majority import input_to_majority_vote_input


@pytest.fixture
def prob_values():
    num_classes = 2
    z = np.zeros((3, 4))
    t = np.zeros((4, 2))

    x = np.random.randint(3, size=(3, 5))
    x_dataset = TensorDataset(torch.Tensor(x))

    z[0, 0] = 1
    z[0, 2] = 1
    z[1, 0] = 1
    z[1, 1] = 1
    z[1, 2] = 1
    z[1, 3] = 1
    z[2, 1] = 1

    t[0, 0] = 1
    t[1, 1] = 1
    t[2, 1] = 1
    t[3, 1] = 1

    model = LogisticRegressionModel(x.shape[1], num_classes)
    config = CleanLabConfig(cv_n_folds=3, psx_calculation_method="rules", seed=123)
    x_dataset, y, z = input_to_majority_vote_input(z, t, x_dataset, use_probabilistic_labels=False)

    trainer = CleanLabTrainer(
        model=model,
        mapping_rules_labels_t=t,
        model_input_x=x_dataset,
        rule_matches_z=z,
        trainer_config=config
    )

    gold_train_dataset = TensorDataset(torch.Tensor(x[2:, :]), torch.Tensor(y[2:]))
    gold_test_dataset = TensorDataset(torch.Tensor(x[:2, :]), torch.Tensor(y[:2]))

    # todo: it all doesn't work!
    # all_gold_train_dataset = [
    #     TensorDataset(torch.Tensor(x[2:, :]), torch.Tensor(y[2:])),
    #     TensorDataset(torch.Tensor(x[0, :]), torch.Tensor(np.array(([0])))),
    #     TensorDataset(torch.Tensor(x[2:, :]), torch.Tensor(y[2:])),
    #     TensorDataset(torch.Tensor(np.stack((x[0, :], x[2, :]))), torch.Tensor(np.array((y[0], y[2])))),
    # ]
    # all_gold_test_dataset = [
    #     TensorDataset(torch.Tensor(x[:2, :]), torch.Tensor(y[:2])),
    #     TensorDataset(torch.Tensor(x[1:, :]), torch.Tensor(y[1:])),
    #     TensorDataset(torch.Tensor(x[:2, :]), torch.Tensor(y[:2])),
    #     TensorDataset(torch.Tensor(x[1, :]), torch.Tensor(y[1])),
    # ]

    return trainer, y, gold_train_dataset, gold_test_dataset,   # all_gold_train_dataset, all_gold_test_dataset


# def test_calculate_psx(prob_values):
#     trainer = prob_values
#     return


def test_get_train_test_datasets_by_rule_indices(prob_values):
    trainer, noisy_y_train, gold_train_dataset, gold_test_dataset = prob_values

    random.seed(trainer.trainer_config.seed)

    train_dataset, test_dataset = get_train_test_datasets_by_rule_indices(
        data_features=trainer.model_input_x,
        rules_ids=[0, 1, 2, 3],
        rule2samples={0: [0, 1], 1: [1, 2], 2: [0, 1], 3: [1]},
        labels=noisy_y_train,
        fold_id=0,
        num_folds=trainer.trainer_config.cv_n_folds
    )

    assert torch.all(train_dataset.tensors[0].eq(gold_train_dataset.tensors[0]))
    assert torch.all(test_dataset.tensors[0].eq(gold_test_dataset.tensors[0]))
    assert torch.all(test_dataset.tensors[1].eq(gold_test_dataset.tensors[1]))


def test_k_folds_splitting_by_rules(prob_values):
    trainer, noisy_y_train, _, _ = prob_values

    train_datasets, test_datasets = k_folds_splitting_by_rules(
        data_features=trainer.model_input_x,
        labels=noisy_y_train,
        rule_matches_z=trainer.rule_matches_z,
        partitions=1,
        num_folds=trainer.trainer_config.cv_n_folds,
        seed=trainer.trainer_config.seed
    )

    random_dataset = test_datasets[random.randint(0, len(test_datasets) - 1)].tensors
    random_idx = random.randint(0, random_dataset[0].shape[0] - 1)
    sample = random_dataset[0][random_idx]
    idx = random_dataset[1][random_idx]

    assert torch.all(trainer.model_input_x.tensors[0][idx].eq(sample))

# todo: test of k_folds_splitting_by_signatures
