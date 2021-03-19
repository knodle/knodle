import joblib
import os

import numpy as np
from snorkel.labeling.model import LabelModel

from torch.optim import SGD
from torch.utils.data import TensorDataset

from knodle.transformation.torch_input import input_labels_to_tensordataset
from knodle.transformation.filter import filter_empty_probabilities

from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.baseline.no_denoising import NoDenoisingTrainer
from knodle.trainer.knn_denoising.knn_denoising import KnnDenoisingTrainer

from knodle.trainer.snorkel.config import SnorkelConfig, SnorkelKNNConfig
from knodle.trainer.snorkel.utils import z_t_matrix_to_snorkel_matrix


@AutoTrainer.register('snorkel')
class SnorkelTrainer(NoDenoisingTrainer):
    def __init__(self, **kwargs):
        if kwargs.get("trainer_config", None) is None:
            kwargs["trainer_config"] = SnorkelConfig(optimizer=SGD(kwargs.get("model").parameters(), lr=0.001))
        super().__init__(**kwargs)

    def _snorkel_denoising(self):
        if self.trainer_config.other_class_id is not None and self.trainer_config.filter_non_labelled:
            raise ValueError("You can either filter samples with no weak labels or add them to 'other_class_id'")

        if self.trainer_config.filter_non_labelled:
            # filter instance with no LF matches
            model_input_x, rule_matches_z = filter_empty_probabilities(self.model_input_x, filter_matrix=self.rule_matches_z)
        else:
            model_input_x = self.model_input_x
            rule_matches_z = self.rule_matches_z

        # snorkel has no benefit from zero rows; they are excluded from training and prediction
        non_zero_indices = np.where(self.rule_matches_z.sum(axis=1) != 0)[0]
        L_train = z_t_matrix_to_snorkel_matrix(rule_matches_z[non_zero_indices], self.mapping_rules_labels_t)

        # train LabelModel
        label_model = LabelModel(cardinality=self.mapping_rules_labels_t.shape[1], verbose=True)
        label_model.fit(
            L_train,
            n_epochs=self.trainer_config.label_model_num_epochs,
            log_freq=self.trainer_config.label_model_log_freq,
            seed=self.trainer_config.seed
        )
        label_probs = label_model.predict_proba(L_train)

        if self.trainer_config.other_class_id:
            # post-process snorkel labels to have other class id
            label_probs_full = np.zeros((rule_matches_z.shape[0], self.trainer_config.output_classes))

            # get ids of all base classes (not other_class_id)
            non_other_class_labels = np.where(
                np.arange(self.trainer_config.output_classes) != self.trainer_config.other_class_id)[0]

            # unusual subsetting: take only specific rows and only specific column values
            label_probs_full[
                non_zero_indices.reshape(-1, 1),
                non_other_class_labels.reshape(1, -1)
            ] = label_probs

            # set rows with no prediction to other class
            label_probs_full[~non_zero_indices, self.trainer_config.other_class_id] = 1.0
            label_probs = label_probs_full

        # cache the resulting labels
        if self.trainer_config.caching_folder is not None:
            cache_dir = self.trainer_config.caching_folder
            cache_file = os.path.join(cache_dir, "snorkel_labels.lib")
            os.makedirs(cache_dir, exist_ok=True)
            joblib.dump(label_probs, cache_file)

        return model_input_x, label_probs

    def train(self):
        # Snorkel denoising
        model_input_x, label_probs = self._snorkel_denoising()

        # Standard training
        feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self.train_loop(feature_label_dataloader)


@AutoTrainer.register('snorkel_knn')
class SnorkelKNNDenoisingTrainer(SnorkelTrainer, KnnDenoisingTrainer):
    def __init__(self, **kwargs):
        if kwargs.get("trainer_config", None) is None:
            kwargs["trainer_config"] = SnorkelKNNConfig(optimizer_=SGD(kwargs.get("model").parameters(), lr=0.001))
        super().__init__(**kwargs)

    def train(self):
        # Snorkel denoising
        self._knn_denoise_rule_matches()
        model_input_x, label_probs = self._snorkel_denoising()

        # Standard training
        feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self.train_loop(feature_label_dataloader)
