import copy
import logging
from typing import Union, List, Tuple

import numpy as np
import torch
from cleanlab.latent_estimation import estimate_latent, compute_confident_joint
from sklearn.model_selection import StratifiedKFold
from skorch import NeuralNetClassifier
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from knodle.trainer import MajorityConfig
from knodle.trainer.cleanlab.utils import load_batch
from knodle.trainer.utils.split import (
    k_folds_splitting_by_rules, k_folds_splitting_by_signatures, k_folds_splitting_random, get_dataset_by_sample_ids
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
    # logger.info(config.other_class_id)

    # calculate psx in advance with splitting by rules
    if config.psx_calculation_method == "rules":
        cv_train_datasets, cv_holdout_datasets = k_folds_splitting_by_rules(
            model_input_x,
            noisy_labels,
            rule_matches_z,
            partitions=1,
            num_folds=config.cv_n_folds,
            seed=config.seed,
            other_class_id=config.other_class_id,
            other_coeff=config.other_coeff,
            verbose=config.verbose
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
            other_class_id=config.other_class_id,
            other_coeff=config.other_coeff,
            verbose=config.verbose
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


def estimate_py_noise_matrices_and_cv_pred_proba(
    data_features: TensorDataset,
    labels: np.ndarray,
    model: nn.Module,
    config: MajorityConfig,
    thresholds: Union[List, np.ndarray] = None
):
    """
    Args:
        data_features: torch.TensorDataset with input features
        labels: np.array with noisy labels
        model: PyTorch compatible model
        config: MajorityConfig with parameters for model training
        thresholds: iterable (list or np.array) of shape (K, 1)  or (K,) P(s^=k|s=k)
    Output:
        Returns a tuple of five numpy array matrices in the form: (py, noise_matrix, inverse_noise_matrix,
        joint count matrix i.e. confident joint, predicted probability matrix).
    """

    confident_joint, psx = estimate_confident_joint_and_cv_pred_proba(
        data_features, labels, model, config, thresholds=thresholds
    )

    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        s=labels,
        py_method=config.py_method,
        converge_latent_estimates=config.converge_latent_estimates,
    )

    return py, noise_matrix, inv_noise_matrix, confident_joint, psx


def estimate_confident_joint_and_cv_pred_proba(
    data_features: TensorDataset,
    labels: np.ndarray,
    model: nn.Module,
    config: MajorityConfig,
    thresholds: Union[List, np.ndarray] = None,
    calibrate: bool = True,
) -> Tuple[np.array, np.array]:
    """
    Args:
        data_features : torch.TensorDataset with input features
        labels : np.array with noisy labels
        model : PyTorch compatible model
        config: MajorityConfig with parameters for model training
        thresholds : iterable (list or np.array) of shape (K, 1)  or (K,) P(s^=k|s=k)
        calibrate : boolean parameter whether the confident joint matrix should be calibrated or not
    Returns:
        Returns a tuple of two numpy array matrices in the form: (joint counts matrix, predicted probability matrix)
    """

    # Ensure labels are of type np.array()
    labels = np.asarray(labels)

    # Create cross-validation object for out-of-sample predicted probabilities.
    kf = StratifiedKFold(n_splits=config.cv_n_folds, shuffle=True, random_state=config.seed)

    # Intialize psx array
    psx = np.zeros((len(labels), config.output_classes))

    # Split X and s into "cv_n_folds" stratified folds.
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(data_features, labels)):

        model_copy = copy.deepcopy(model).to(config.device)

        # Select the training and holdout cross-validated sets.
        train_dataset = get_dataset_by_sample_ids(data_features, labels, cv_train_idx, save_ids=False)
        test_dataset = get_dataset_by_sample_ids(data_features, labels, cv_holdout_idx, save_ids=True)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

        # Train the clf classifier to the training set and predict on the holdout set and update psx.
        model_copy = wscl_train_loop(model_copy, train_loader, config)  # todo: add special params for calculation psx
        psx_cv = wscl_test_loop(model_copy, test_loader, config)
        psx[cv_holdout_idx] = psx_cv.cpu().detach().numpy()

    # Compute the confident counts, a K x K matrix for all pairs of labels.
    confident_joint = compute_confident_joint(
        s=labels,
        psx=psx,  # P(s = k|x)
        thresholds=thresholds,
        calibrate=calibrate,
    )

    return confident_joint, psx


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
