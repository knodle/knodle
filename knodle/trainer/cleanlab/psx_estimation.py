import copy
from typing import Union

import numpy as np
from cleanlab.latent_estimation import estimate_cv_predicted_probabilities
from sklearn.base import RegressorMixin
from skorch import NeuralNetClassifier
from torch.utils.data import TensorDataset
from tqdm import tqdm

from knodle.trainer.wscrossweigh.data_splitting_by_rules import (
    k_folds_splitting_by_rules, k_folds_splitting_by_signatures
)
from knodle.transformation.torch_input import dataset_to_numpy_input


def calculate_psx(
        model_input_x: TensorDataset,
        noisy_labels: np.ndarray,
        rule_matches_z: np.ndarray,
        model: NeuralNetClassifier,
        psx_calculation_method: str,
        num_classes: int,
        cv_n_folds: int,
        seed: int
) -> Union[np.ndarray, None]:

    # calculate psx in advance with splitting by rules
    if psx_calculation_method == "rules":
        return estimate_cv_predicted_probabilities_split_by_rules(
            model_input_x, noisy_labels, rule_matches_z, model, num_classes, cv_n_folds=cv_n_folds, seed=seed
        )

    # calculate psx in advance with splitting by signatures
    elif psx_calculation_method == "signatures":
        return estimate_cv_predicted_probabilities_split_by_signatures(
            model_input_x, noisy_labels, rule_matches_z, model, num_classes, cv_n_folds=cv_n_folds, seed=seed
        )

    # calculate psx in advance with splitting randomly
    elif psx_calculation_method == "random":
        # if no special psx calculation method is specified, psx will be calculated with random folder splitting
        return estimate_cv_predicted_probabilities_split_randomly(
            model_input_x, noisy_labels, model, cv_n_folds=cv_n_folds, seed=seed
        )

    else:
        raise ValueError("Unknown psx calculation method.")


def estimate_cv_predicted_probabilities_split_by_rules(
        model_input_x: TensorDataset,
        noisy_labels: np.ndarray,
        rule_matches_z: np.ndarray,
        model: NeuralNetClassifier,
        num_classes: int,
        cv_n_folds: int = 5,
        seed: int = None,
        other_class_id: int = None
):
    cv_train_datasets, cv_holdout_datasets = k_folds_splitting_by_rules(
        model_input_x,
        noisy_labels,
        rule_matches_z,
        partitions=1,
        num_folds=cv_n_folds,
        seed=seed,
        other_class_id=other_class_id
    )

    return compute_psx_matrix(model, cv_train_datasets, cv_holdout_datasets, noisy_labels, num_classes)


def estimate_cv_predicted_probabilities_split_by_signatures(
        model_input_x: TensorDataset,
        noisy_labels: np.ndarray,
        rule_matches_z: np.ndarray,
        model: NeuralNetClassifier,
        num_classes: int,
        cv_n_folds: int = 5,
        seed: int = None,
        other_class_id: int = None
) -> np.ndarray:

    cv_train_datasets, cv_holdout_datasets = k_folds_splitting_by_signatures(
        model_input_x,
        noisy_labels,
        rule_matches_z,
        partitions=1,
        num_folds=cv_n_folds,
        seed=seed,
        other_class_id=other_class_id
    )

    return compute_psx_matrix(model, cv_train_datasets, cv_holdout_datasets, noisy_labels, num_classes)


def estimate_cv_predicted_probabilities_split_randomly(
        model_input_x: TensorDataset,
        noisy_labels: np.ndarray,
        model: RegressorMixin,
        cv_n_folds: int = 5,
        seed: int = None,
):
    # turn input to the CL-compatible format
    model_input_x_numpy = dataset_to_numpy_input(model_input_x)

    # inherited from the original CL
    return estimate_cv_predicted_probabilities(
        X=model_input_x_numpy,
        labels=noisy_labels,
        clf=model,
        cv_n_folds=cv_n_folds,
        seed=seed,
    )


def compute_psx_matrix(
        model: NeuralNetClassifier,
        cv_train_datasets: TensorDataset,
        cv_holdout_datasets: TensorDataset,
        labels: np.ndarray,
        num_classes: int
) -> np.ndarray:

    psx = np.zeros((len(labels), num_classes))

    for k, (cv_train_dataset, cv_holdout_dataset) in tqdm(enumerate(zip(cv_train_datasets, cv_holdout_datasets))):
        model_copy = copy.deepcopy(model)
        X_train_cv, y_train_cv = cv_train_dataset.tensors[0].cpu().detach().numpy(), \
                                 cv_train_dataset.tensors[1].cpu().detach().numpy()
        X_holdout_cv, indices_holdout_cv, y_holdout_cv = cv_holdout_dataset.tensors[0].cpu().detach().numpy(), \
                                                         cv_holdout_dataset.tensors[1].cpu().detach().numpy(), \
                                                         cv_holdout_dataset.tensors[2].cpu().detach().numpy()

        # y_train_cv = np.argmax(y_train_cv, axis=1)
        model_copy.fit(X_train_cv, y_train_cv)
        psx_cv = model_copy.predict_proba(X_holdout_cv)  # P(s = k|x) # [:,1]
        psx[indices_holdout_cv] = psx_cv

    return psx
