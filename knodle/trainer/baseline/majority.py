import logging

import numpy as np

import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import TensorDataset

from knodle.transformation.majority import input_to_majority_vote_input
from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.trainer import BaseTrainer
from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.baseline.config import MajorityConfig
from knodle.transformation.filter import filter_unconclusive_probabilities

logger = logging.getLogger(__name__)


@AutoTrainer.register('majority')
class MajorityVoteTrainer(BaseTrainer):
    """
    The baseline class implements a baseline model for labeling data with weak supervision.
        A simple majority vote is used for this purpose.
    """

    def __init__(
            self,
            model: nn.Module,
            mapping_rules_labels_t: np.ndarray,
            model_input_x: TensorDataset,
            rule_matches_z: np.ndarray,
            trainer_config: MajorityConfig = None,
            **kwargs
    ):
        if trainer_config is None:
            trainer_config = MajorityConfig(optimizer=SGD, lr=0.001)
        super().__init__(
            model, mapping_rules_labels_t, model_input_x, rule_matches_z, trainer_config=trainer_config, **kwargs
        )

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None,
            probability_threshold: float = None
    ):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """
        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)

        # initialise optimizer
        self.trainer_config.optimizer = self.initialise_optimizer()

        model_input_x, label_probs = input_to_majority_vote_input(
            self.model_input_x, self.rule_matches_z, self.mapping_rules_labels_t,
            filter_non_labelled=self.trainer_config.filter_non_labelled,
            other_class_id=self.trainer_config.other_class_id
        )

        if self.trainer_config.probability_threshold is not None:
            model_input_x, label_probs = filter_unconclusive_probabilities(
                model_input_x, label_probs, probability_threshold=self.trainer_config.probability_threshold
            )

        feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self._train_loop(feature_label_dataloader)
