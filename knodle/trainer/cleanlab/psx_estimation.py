import copy
import logging
from typing import Union, List

import numpy as np
import torch
from skorch import NeuralNetClassifier
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from knodle.trainer import MajorityConfig
from knodle.trainer.wscrossweigh.data_splitting_by_rules import (
    k_folds_splitting_by_rules, k_folds_splitting_by_signatures, k_folds_splitting_random
)


logger = logging.getLogger(__name__)


def calculate_psx(
        model_input_x: TensorDataset,
        noisy_labels: np.ndarray,
        rule_matches_z: np.ndarray,
        model: NeuralNetClassifier,
        config: MajorityConfig,
) -> Union[np.ndarray, None]:

    num_samples = len(model_input_x)

    # calculate psx in advance with splitting by rules
    if config.psx_calculation_method == "rules":

        cv_train_datasets, cv_holdout_datasets = k_folds_splitting_by_rules(
            model_input_x,
            noisy_labels,
            rule_matches_z,
            partitions=1,
            num_folds=config.cv_n_folds,
            seed=config.seed,
            other_class_id=None
        )

        return compute_psx_matrix(model, cv_train_datasets, cv_holdout_datasets, num_samples, config)

    # calculate psx in advance with splitting by signatures
    elif config.psx_calculation_method == "signatures":
        cv_train_datasets, cv_holdout_datasets = k_folds_splitting_by_signatures(
            model_input_x,
            noisy_labels,
            rule_matches_z,
            partitions=1,
            num_folds=config.cv_n_folds,
            seed=config.seed,
            other_class_id=None
        )

        return compute_psx_matrix(model, cv_train_datasets, cv_holdout_datasets, num_samples, config)

    # calculate psx in advance with splitting randomly
    elif config.psx_calculation_method == "random":
        # if no special psx calculation method is specified, psx will be calculated with random folder splitting
        cv_train_datasets, cv_holdout_datasets = k_folds_splitting_random(
            model_input_x,
            noisy_labels,
            num_folds=config.cv_n_folds,
            seed=config.seed
        )

        return compute_psx_matrix(model, cv_train_datasets, cv_holdout_datasets, num_samples, config)

    else:
        raise ValueError("Unknown psx calculation method.")


def compute_psx_matrix(
        model: NeuralNetClassifier,
        cv_train_datasets: List[TensorDataset],
        cv_holdout_datasets: List[TensorDataset],
        num_samples: int,
        config: MajorityConfig
) -> np.ndarray:
    """ my function """
    psx = np.zeros((num_samples, config.output_classes))

    for k, (cv_train_dataset, cv_holdout_dataset) in tqdm(enumerate(zip(cv_train_datasets, cv_holdout_datasets))):
        indices_holdout_cv = cv_holdout_dataset.tensors[1].cpu().detach().numpy()
        curr_psx_cv = calculate_psx_row(cv_train_dataset, cv_holdout_dataset, model, config)

        assert [psx[idx][0] == 0 and psx[idx][1] == 0 for idx in indices_holdout_cv]

        psx[indices_holdout_cv] = curr_psx_cv.cpu().detach().numpy()

    return psx


def calculate_psx_row(
        cv_train_dataset: TensorDataset, cv_holdout_dataset: TensorDataset, model: NeuralNetClassifier,
        config: MajorityConfig
) -> Tensor:
    model_copy = copy.deepcopy(model).to(config.device)

    test_loader = DataLoader(cv_holdout_dataset, batch_size=config.batch_size)
    train_loader = DataLoader(cv_train_dataset, batch_size=config.batch_size)

    model_copy = wscl_train_loop(model_copy, train_loader, config)  # todo: add special params for calculation psx
    psx_cv = wscl_test_loop(model_copy, test_loader, config)

    return psx_cv


def wscl_train_loop(model: nn.Module, feature_label_dataloader: DataLoader, config: MajorityConfig):
    model.to(config.device)
    model.train()

    optimizer = config.psx_optimizer(params=model.parameters(), lr=config.psx_lr)

    for current_epoch in range(config.psx_epochs):

        for batch in feature_label_dataloader:
            features, labels = load_batch(batch, config.device)

            # forward pass
            optimizer.zero_grad()
            outputs = model(*features)
            if isinstance(outputs, Tensor):
                logits = outputs
            else:
                logits = outputs[0]

            # todo: duplicated code (the same is in trainer.py)
            if isinstance(config.psx_criterion, type) and issubclass(config.psx_criterion, _Loss):
                criterion = config.psx_criterion(weight=config.class_weights).to(config.device)
                loss = criterion(logits, labels)
            else:
                loss = config.psx_criterion(logits, labels, weight=config.class_weights)

            if isinstance(config.grad_clipping, (int, float)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)

            loss.backward()
            optimizer.step()
            model.to(config.device)
    return model


def wscl_test_loop(model: nn.Module, test_loader: DataLoader, config: MajorityConfig) -> Tensor:
    model.eval()
    predictions_list = []

    with torch.no_grad():
        for batch in test_loader:
            features, labels = load_batch(batch, config.device)
            data_features, data_indices = features[:-1], features[-1]

            outputs = model(*data_features)
            prediction_vals = outputs[0] if not isinstance(outputs, torch.Tensor) else outputs
            predictions_list.append(prediction_vals)
    predictions = torch.cat(predictions_list)
    return predictions


def load_batch(batch, device):
    features = [inp.to(device) for inp in batch[0: -1]]
    labels = batch[-1].to(device)
    return features, labels
