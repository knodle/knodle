import logging

import numpy as np
from torch.optim import SGD
from torch.utils.data import TensorDataset

from knodle.transformation.majority import input_to_majority_vote_input
from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.trainer import BaseTrainer
from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.baseline.config import MajorityConfig

logger = logging.getLogger(__name__)


@AutoTrainer.register('majority')
class MajorityVoteTrainer(BaseTrainer):
    """
    The baseline class implements a baseline model for labeling data with weak supervision.
        A simple majority vote is used for this purpose.
    """

    def __init__(self, **kwargs):
        if kwargs.get("trainer_config") is None:
            kwargs["trainer_config"] = MajorityConfig(optimizer=SGD, lr=0.001)
        super().__init__(**kwargs)

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """
        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)
        self._apply_rule_reduction()

        # initialise optimizer
        self.trainer_config.optimizer = self.initialise_optimizer()

        self.model_input_x, noisy_y_train, self.rule_matches_z = input_to_majority_vote_input(
            self.rule_matches_z, self.mapping_rules_labels_t, self.model_input_x,
            probability_threshold=self.trainer_config.probability_threshold,
            unmatched_strategy=self.trainer_config.unmatched_strategy,
            ties_strategy=self.trainer_config.ties_strategy,
            use_probabilistic_labels=self.trainer_config.use_probabilistic_labels,
            other_class_id=self.trainer_config.other_class_id,
            multi_label=self.trainer_config.multi_label,
            multi_label_threshold=self.trainer_config.multi_label_threshold
        )

        if self.trainer_config.use_probabilistic_labels:
            feature_label_dataset = input_labels_to_tensordataset(self.model_input_x, noisy_y_train, probs=True)
        else:
            feature_label_dataset = input_labels_to_tensordataset(self.model_input_x, noisy_y_train, probs=False)

        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self._train_loop(feature_label_dataloader)

