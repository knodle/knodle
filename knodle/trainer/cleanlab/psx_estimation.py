import copy
import logging
from typing import Union

import numpy as np
import torch
from skorch import NeuralNetClassifier
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from knodle.trainer.wscrossweigh.data_splitting_by_rules import (
    k_folds_splitting_by_rules, k_folds_splitting_by_signatures, k_folds_splitting_random
)


logger = logging.getLogger(__name__)


def calculate_psx(
        model_input_x: TensorDataset,
        noisy_labels: np.ndarray,
        rule_matches_z: np.ndarray,
        model: NeuralNetClassifier,
        config,
) -> Union[np.ndarray, None]:

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

        return compute_psx_matrix(model, cv_train_datasets, cv_holdout_datasets, noisy_labels, config)

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

        return compute_psx_matrix(model, cv_train_datasets, cv_holdout_datasets, noisy_labels, config)

    # calculate psx in advance with splitting randomly
    elif config.psx_calculation_method == "random":
        # if no special psx calculation method is specified, psx will be calculated with random folder splitting
        cv_train_datasets, cv_holdout_datasets = k_folds_splitting_random(
            model_input_x,
            noisy_labels,
            num_folds=config.cv_n_folds,
            seed=config.seed
        )

        return compute_psx_matrix(model, cv_train_datasets, cv_holdout_datasets, noisy_labels, config)

    else:
        raise ValueError("Unknown psx calculation method.")


def compute_psx_matrix(
        model: NeuralNetClassifier,
        cv_train_datasets: TensorDataset,
        cv_holdout_datasets: TensorDataset,
        labels: np.ndarray,
        config
) -> np.ndarray:
    """ my function """

    if not config.output_classes:
        output_classes = len(np.unique(labels))
    else:
        output_classes = config.output_classes

    # Ensure labels are of type np.array()
    labels = np.asarray(labels)

    psx = np.zeros((len(labels), output_classes))

    for k, (cv_train_dataset, cv_holdout_dataset) in tqdm(enumerate(zip(cv_train_datasets, cv_holdout_datasets))):
        model_copy = copy.deepcopy(model).to(config.device)

        indices_holdout_cv = cv_holdout_dataset.tensors[1].cpu().detach().numpy()

        test_loader = make_dataloader(cv_holdout_dataset)   # todo: here shuffle = True by default. Shouldn't it be false?
        train_loader = make_dataloader(cv_train_dataset)

        model_copy = wscl_train_loop(
            model_copy,
            train_loader,
            config      # todo: add special params for calculation psx
        )
        psx_cv = wscl_test_loop(
            model_copy,
            test_loader,
            config.device
        )
        psx[indices_holdout_cv] = psx_cv

        # Fit the clf classifier to the training set and
        # predict on the holdout set and update psx.
        # model_copy.fit(X_train_cv, y_train_cv)
        # psx_cv = model_copy.predict_proba(X_holdout_cv)  # P(s = k|x) # [:,1]
        # psx[indices_holdout_cv] = psx_cv

    return psx


def wscl_train_loop(model, feature_label_dataloader, config):
    model.to(config.device)
    model.train()

    optimizer = config.optimizer(params=model.parameters(), lr=config.lr)
    criterion = config.criterion()

    for current_epoch in range(config.epochs):
        logger.info("Epoch: {}".format(current_epoch))

        for batch in tqdm(feature_label_dataloader):
            features, labels = load_batch(batch, config.device)

            # forward pass
            optimizer.zero_grad()
            outputs = model(*features)
            if isinstance(outputs, Tensor):
                logits = outputs
            else:
                logits = outputs[0]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    return model


def wscl_test_loop(model, test_loader, device):
    model.eval()
    predictions_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            features, labels = load_batch(batch, device)
            data_features, data_indices = features[:-1], features[-1]

            outputs = model(*data_features)
            prediction_vals = outputs[0] if not isinstance(outputs, torch.Tensor) else outputs
            predictions_list.append(prediction_vals)
    predictions = np.squeeze(np.hstack(predictions_list))
    return predictions


def load_batch(batch, device):
    features = [inp.to(device) for inp in batch[0: -1]]
    labels = batch[-1].to(device)
    return features, labels


def make_dataloader(dataset: TensorDataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=shuffle)
