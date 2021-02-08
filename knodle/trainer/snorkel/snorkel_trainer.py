import scipy.sparse as sp
import numpy as np
from snorkel.labeling.model import LabelModel

from torch.utils.data import TensorDataset

from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.baseline.mixed import MajorityTrainer


def z_t_matrix_to_snorkel_matrix(rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray):
    snorkel_matrix = -1 * np.ones(rule_matches_z.shape)

    if isinstance(rule_matches_z, sp.csr_matrix):
        rule_matches_z = rule_matches_z.toarray()

    z_to_t = np.argmax(mapping_rules_labels_t, axis=-1)
    for i in range(rule_matches_z.shape[0]):
        non_zero_idx = np.where(rule_matches_z[i] != 0)[0]
        snorkel_matrix[i, non_zero_idx] = z_to_t[non_zero_idx]

    return snorkel_matrix


class SnorkelTrainer(MajorityTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):
        non_zero_indices = np.where(self.rule_matches_z.sum(axis=1) != 0)[0]
        rule_matches_z = self.rule_matches_z[non_zero_indices]
        tensors = list(self.model_input_x.tensors)
        for i in range(len(tensors)):
            tensors[i] = tensors[i][non_zero_indices]

        model_input_x = TensorDataset(*tensors)

        L_train = z_t_matrix_to_snorkel_matrix(rule_matches_z, self.mapping_rules_labels_t)

        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)  # TODO values

        probs_train = label_model.predict_proba(L_train)

        feature_label_dataset = input_labels_to_tensordataset(model_input_x, probs_train)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self.train_loop(feature_label_dataloader)
