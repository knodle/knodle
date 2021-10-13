import copy
import logging
from typing import Union, List

import numpy as np
import torch
from cleanlab.latent_estimation import estimate_latent, compute_confident_joint
from cleanlab.util import assert_inputs_are_valid
from sklearn.model_selection import StratifiedKFold
from skorch import NeuralNetClassifier
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from knodle.trainer import MajorityConfig
from knodle.trainer.wscrossweigh.data_splitting_by_rules import (
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


def estimate_py_noise_matrices_and_cv_pred_proba_baseline(
    data_features,
    labels,
    model,
    config,
    cv_n_folds=5,
    thresholds=None,
    converge_latent_estimates=False,
    py_method='cnt',
    seed=None,
):
    """This function computes the out-of-sample predicted
    probability P(s=k|x) for every example x in X using cross
    validation while also computing the confident counts noise
    rates within each cross-validated subset and returning
    the average noise rate across all examples.

    This function estimates the noise_matrix of shape (K, K). This is the
    fraction of examples in every class, labeled as every other class. The
    noise_matrix is a conditional probability matrix for P(s=k_s|y=k_y).

    Under certain conditions, estimates are exact, and in most
    conditions, estimates are within one percent of the actual noise rates.

    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array

    s : np.array
        A discrete vector of labels, s, which may contain mislabeling. "s"
        denotes the noisy label instead of \tilde(y), for ASCII reasons.

    clf : sklearn.classifier or equivalent
      Default classifier used is logistic regression. Assumes clf
      has predict_proba() and fit() defined.

    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
      P(s^=k|s=k). If an example has a predicted probability "greater" than
      this threshold, it is counted as having hidden label y = k. This is
      not used for pruning, only for estimating the noise rates using
      confident counts. This value should be between 0 and 1. Default is None.

    converge_latent_estimates : bool
      If true, forces numerical consistency of estimates. Each is estimated
      independently, but they are related mathematically with closed form
      equivalences. This will iteratively make them mathematically consistent.

    py_method : str (Options: ["cnt", "eqn", "marginal", "marginal_ps"])
        How to compute the latent prior p(y=k). Default is "cnt" as it often
        works well even when the noise matrices are estimated poorly by using
        the matrix diagonals instead of all the probabilities.

    seed : int (default = None)
        Set the default state of the random number generator used to split
        the cross-validated folds. If None, uses np.random current random state.

    Returns
    ------
      Returns a tuple of five numpy array matrices in the form:
      (py, noise_matrix, inverse_noise_matrix,
      joint count matrix i.e. confident joint, predicted probability matrix)"""

    confident_joint, psx = estimate_confident_joint_and_cv_pred_proba_baseline(
        data_features, labels, model, config,
        cv_n_folds=cv_n_folds,
        thresholds=thresholds,
        seed=seed,
    )

    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        s=labels,
        py_method=py_method,
        converge_latent_estimates=converge_latent_estimates,
    )

    return py, noise_matrix, inv_noise_matrix, confident_joint, psx


def estimate_confident_joint_and_cv_pred_proba_baseline(
    data_features,
    labels,
    model,
    config,
    cv_n_folds=5,
    thresholds=None,
    seed=None,
    calibrate=True,
):
    """Estimates P(s,y), the confident counts of the latent
    joint distribution of true and noisy labels
    using observed s and predicted probabilities psx.

    The output of this function is a numpy array of shape (K, K).

    Under certain conditions, estimates are exact, and in many
    conditions, estimates are within one percent of actual.

    Notes: There are two ways to compute the confident joint with pros/cons.
    1. For each holdout set, we compute the confident joint, then sum them up.
    2. Compute pred_proba for each fold, combine, compute the confident joint.
    (1) is more accurate because it correctly computes thresholds for each fold
    (2) is more accurate when you have only a little data because it computes
    the confident joint using all the probabilities. For example if you had 100
    examples, with 5-fold cross validation + uniform p(y) you would only have 20
    examples to compute each confident joint for (1). Such small amounts of data
    is bound to result in estimation errors. For this reason, we implement (2),
    but we implement (1) as a commented out function at the end of this file.

    Parameters
    ----------
    X : np.array
      Input feature matrix (N, D), 2D numpy array

    s : np.array
        A discrete vector of labels, s, which may contain mislabeling. "s"
        denotes the noisy label instead of \tilde(y), for ASCII reasons.

    clf : sklearn.classifier or equivalent
      Default classifier used is logistic regression. Assumes clf
      has predict_proba() and fit() defined.

    cv_n_folds : int
      The number of cross-validation folds used to compute
      out-of-sample probabilities for each example in X.

    thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
      P(s^=k|s=k). If an example has a predicted probability "greater" than
      this threshold, it is counted as having hidden label y = k. This is
      not used for pruning, only for estimating the noise rates using
      confident counts. This value should be between 0 and 1. Default is None.

    seed : int (default = None)
        Set the default state of the random number generator used to split
        the cross-validated folds. If None, uses np.random current random state.

    calibrate : bool (default: True)
        Calibrates confident joint estimate P(s=i, y=j) such that
        np.sum(cj) == len(s) and np.sum(cj, axis = 1) == np.bincount(s).

    Returns
    ------
      Returns a tuple of two numpy array matrices in the form:
      (joint counts matrix, predicted probability matrix)"""

    # Ensure labels are of type np.array()
    labels = np.asarray(labels)

    # Create cross-validation object for out-of-sample predicted probabilities.
    # CV folds preserve the fraction of noisy positive and
    # noisy negative examples in each class.
    kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=seed)

    # Intialize psx array
    psx = np.zeros((len(labels), config.output_classes))

    # Split X and s into "cv_n_folds" stratified folds.
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(data_features, labels)):

        model_copy = copy.deepcopy(model).to(config.device)

        # Select the training and holdout cross-validated sets.
        train_dataset = get_dataset_by_sample_ids(data_features, labels, cv_train_idx, save_ids=False)
        test_dataset = get_dataset_by_sample_ids(data_features, labels, cv_holdout_idx, save_ids=True)

        # X_train_cv, X_holdout_cv = X[cv_train_idx], X[cv_holdout_idx]
        # s_train_cv, s_holdout_cv = s[cv_train_idx], s[cv_holdout_idx]

        # Fit the clf classifier to the training set and
        # predict on the holdout set and update psx.

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

        model_copy = wscl_train_loop(model_copy, train_loader, config)  # todo: add special params for calculation psx
        psx_cv = wscl_test_loop(model_copy, test_loader, config)
        psx[cv_holdout_idx] = psx_cv
        #
        #
        # clf.train()
        # clf_copy.fit(X_train_cv, s_train_cv)
        # psx_cv = clf_copy.predict_proba(X_holdout_cv)  # P(s = k|x) # [:,1]
        # psx[cv_holdout_idx] = psx_cv

    # Compute the confident counts, a K x K matrix for all pairs of labels.
    confident_joint = compute_confident_joint(
        s=labels,
        psx=psx,  # P(s = k|x)
        thresholds=thresholds,
        calibrate=calibrate,
    )

    return confident_joint, psx