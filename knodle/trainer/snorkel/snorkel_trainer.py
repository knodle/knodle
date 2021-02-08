import numpy as np
from snorkel.labeling.model import LabelModel

from knodle.trainer.trainer import Trainer


def z_t_matrix_to_snorkel_matrix(rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray):
    snorkel_matrix = -1 * np.ones(rule_matches_z.shape)

    z_to_t = np.argmax(mapping_rules_labels_t, axis=-1)

    for i in range(rule_matches_z.shape[0]):
        non_zero_idx = np.where(rule_matches_z[i] != 0)[0]
        snorkel_matrix[non_zero_idx] = z_to_t[non_zero_idx]

    return snorkel_matrix


class SnorkelTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):

        zero_indices = np.where(self.rule_matches_z.sum(axis=1) == 0)[0]
        rule_matches_z = self.rule_matches_z[zero_indices]
        model_input_x = self.model_input_x[zero_indices]

        L_train = z_t_matrix_to_snorkel_matrix(self.rule_matches_z, self.mapping_rules_labels_t)

        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345) # TODO values

        probs_train = label_model.predict_proba(L_train)

        # prediction loop


