import copy

import numpy as np
from sklearn.base import RegressorMixin
from tqdm import tqdm

from knodle.trainer.crossweigh_weighing.data_preparation import k_folds_splitting


def estimate_cv_predicted_probabilities_split_by_rules(
        model_input_x: np.ndarray,
        labels: np.ndarray,
        rule_matches_z: np.ndarray,
        model: RegressorMixin,
        num_classes: int,
        cv_n_folds: int = 5,
        other_class_id: int = None
):
    cv_train_datasets, cv_holdout_datasets = k_folds_splitting(
        model_input_x, labels, rule_matches_z, partitions=1, folds=cv_n_folds, other_class_id=other_class_id
    )

    psx = np.zeros((len(labels), num_classes))

    filled_indices = []
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
        # for ind in indices_holdout_cv.tolist():
        #     if ind not in filled_indices:
        #         filled_indices.append(ind)
        #     else:
        #         print("ALARMAAAAAAA")

        psx[indices_holdout_cv] = psx_cv  # todo: double triple whatever check - intuition only!

    return psx
