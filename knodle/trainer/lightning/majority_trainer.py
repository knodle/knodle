from typing import Optional

import numpy as np

from knodle.data import KnodleDataModule
from knodle.transformation.majority import z_t_matrices_to_majority_vote_probs
from knodle.trainer.lightning.trainer import KnodleLightningLabelModelTrainer


class LightningMajorityTrainer(KnodleLightningLabelModelTrainer):
    def __init__(
            self,
            use_probabilistic_labels: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.use_probabilistic_labels = use_probabilistic_labels
        self.probability_threshold = None

    def apply_majority_voting(self, datamodule: Optional[KnodleDataModule], debug=True):
        # pass
        noisy_y_train = z_t_matrices_to_majority_vote_probs(
            rule_matches_z=datamodule.data["train_rule_matches_z"],
            mapping_rules_labels_t=datamodule.data["mapping_rules_labels_t"],
            other_class_id=None
        )
        prob_sums = noisy_y_train.sum(axis=-1)
        non_zeros = np.where(prob_sums != 0)[0]
        if debug:
            non_zeros = non_zeros[0:32]

        datamodule.data["train_rule_matches_z"] = datamodule.data["train_rule_matches_z"][non_zeros]
        datamodule.data["train_weak_y"] = noisy_y_train[non_zeros].argmax(axis=-1)

        for key in datamodule.dataloader_train_keys:
            datamodule.data[f"train_{key}"] = datamodule.data[f"train_{key}"][non_zeros]

        assert datamodule.data["train_weak_y"].shape[0] == datamodule.data["train_rule_matches_z"].shape[0]
        return datamodule

    def apply_label_model(self, datamodule):
        return self.apply_majority_voting(datamodule)
