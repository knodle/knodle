import numpy as np
from snorkel.labeling.model import LabelModel

from torch.utils.data import TensorDataset

from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.knn_denoising.knn_denoising import KnnDenoisingTrainer

from knodle.trainer.snorkel.config import SnorkelConfig, SnorkelKNNConfig
from knodle.trainer.snorkel.utils import z_t_matrix_to_snorkel_matrix


class SnorkelTrainer(SnorkelConfig):
    def __init__(self, **kwargs):
        trainer_config = kwargs.get("trainer_config", None)
        if trainer_config is None:
            trainer_config = SnorkelConfig(self.model)
        super().__init__(trainer_config=trainer_config, **kwargs)

    def _snorkel_denoising(self, model_input_x, rule_matches_z):
        non_zero_indices = np.where(rule_matches_z.sum(axis=1) != 0)[0]
        rule_matches_z = rule_matches_z[non_zero_indices]
        tensors = list(model_input_x.tensors)
        for i in range(len(tensors)):
            tensors[i] = tensors[i][non_zero_indices]

        model_input_x = TensorDataset(*tensors)

        # create Snorkel matrix and train LabelModel
        L_train = z_t_matrix_to_snorkel_matrix(rule_matches_z, self.mapping_rules_labels_t)

        label_model = LabelModel(cardinality=self.mapping_rules_labels_t.shape[1], verbose=True)
        label_model.fit(
            L_train,
            n_epochs=self.trainer_config.label_model_num_epochs,
            log_freq=self.trainer_config.label_model_log_freq,
            seed=self.trainer_config.seed
        )
        label_probs = label_model.predict_proba(L_train)
        return model_input_x, label_probs

    def train(self):
        # Snorkel denoising
        model_input_x, label_probs = self._snorkel_denoising(self.model_input_x, self.rule_matches_z)

        # Standard training
        feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self.train_loop(feature_label_dataloader)


class SnorkelKNNDenoisingTrainer(SnorkelTrainer, KnnDenoisingTrainer):
    def __init__(self, **kwargs):
        trainer_config = kwargs.get("trainer_config", None)
        if trainer_config is None:
            trainer_config = SnorkelKNNConfig(self.model)
        super().__init__(trainer_config=trainer_config, **kwargs)

    def train(self):
        # Snorkel denoising
        denoised_rule_matches_z = self._knn_denoise_rule_matches()
        model_input_x, label_probs = self._snorkel_denoising(self.model_input_x, denoised_rule_matches_z)

        # Standard training
        feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self.train_loop(feature_label_dataloader)
