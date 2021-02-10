import numpy as np
from snorkel.labeling.model import LabelModel

from torch.utils.data import TensorDataset

from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.snorkel.snorkel_config import SnorkelConfig
from knodle.trainer.snorkel.utils import z_t_matrix_to_snorkel_matrix


class SnorkelTrainer(SnorkelConfig):
    def __init__(self, **kwargs):
        trainer_config = kwargs.get("trainer_config", None)
        if trainer_config is None:
            trainer_config = SnorkelConfig(self.model)
        super().__init__(trainer_config=trainer_config, **kwargs)

    def train(self):
        # filter empty rows
        non_zero_indices = np.where(self.rule_matches_z.sum(axis=1) != 0)[0]
        rule_matches_z = self.rule_matches_z[non_zero_indices]
        tensors = list(self.model_input_x.tensors)
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
        probs_train = label_model.predict_proba(L_train)

        # Standard training
        feature_label_dataset = input_labels_to_tensordataset(model_input_x, probs_train)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self.train_loop(feature_label_dataloader)
