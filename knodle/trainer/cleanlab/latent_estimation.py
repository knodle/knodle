import copy

import numpy as np
from sklearn.base import RegressorMixin
from tqdm import tqdm

from knodle.trainer.crossweigh_weighing.data_splitting_by_rules import k_folds_splitting_by_rules


def estimate_cv_predicted_probabilities_split_by_rules(
        model_input_x: np.ndarray,
        labels: np.ndarray,
        rule_matches_z: np.ndarray,
        model: RegressorMixin,
        num_classes: int,
        cv_n_folds: int = 5,
        seed: int = None,
        other_class_id: int = None
):
    psx = np.zeros((len(labels), num_classes))

    cv_train_datasets, cv_holdout_datasets = k_folds_splitting_by_rules(
        model_input_x, labels, rule_matches_z, partitions=1, folds=cv_n_folds, seed=seed, other_class_id=other_class_id
    )

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


def estimate_cv_predicted_probabilities_split_by_rules(
        model_input_x: np.ndarray,
        labels: np.ndarray,
        rule_matches_z: np.ndarray,
        model: RegressorMixin,
        num_classes: int,
        cv_n_folds: int = 5,
        other_class_id: int = None
):
    psx = np.zeros((len(labels), num_classes))

    return psx