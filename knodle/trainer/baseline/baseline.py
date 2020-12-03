from torch.nn import Module
from torch import Tensor
import numpy as np

import logging

from knodle.trainer import TrainerConfig
from knodle.trainer.ds_model_trainer.ds_model_trainer import DsModelTrainer
from knodle.trainer.utils import log_section
from knodle.trainer.utils.utils import accuracy_of_probs

logger = logging.getLogger(__name__)


class SimpleDsModelTrainer(DsModelTrainer):
    """
    The baseline class implements a baseline model for labeling data with weak supervision.
        A simple majority vote is used for this purpose.
    """

    def __init__(self, model: Module, trainer_config: TrainerConfig = None):
        super().__init__(model, trainer_config)

    def train(
        self,
        inputs: Tensor,
        rule_matches: np.ndarray,
        mapping_rules_labels: np.ndarray,
        epochs: int,
    ):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        Args:
            inputs: Input tensors. These tensors will be fed to the provided model (instances x features)
            rule_matches: Binary encoded array of which rules matched. Shape: instances x rules
            mapping_rules_labels: Mapping of rules to labels, binary encoded. Shape: rules x classes
            epochs: Epochs to train
        """

        if epochs <= 0:
            raise ValueError("Epochs needs to be positive")

        self.model.train()
        labels = self.get_majority_vote_probs(rule_matches, mapping_rules_labels)

        labels = Tensor(labels)
        log_section("Training starts", logger)

        for current_epoch in range(epochs):
            logger.info("Epoch: {}".format(current_epoch))
            self.model.zero_grad()
            predictions = self.model(inputs)
            loss = self.trainer_config.criterion(predictions, labels)
            logger.info("Loss is: {}".format(loss.detach()))
            loss.backward()
            self.trainer_config.optimizer.step()

        log_section("Training done", logger)

    def get_majority_vote_probs(
        self, rule_matches: np.ndarray, mapping_rules_labels: np.ndarray
    ):
        """
        This function calculates a majority vote probability for all rule_matches. First rule counts will be calculated,
        then a probability will be calculated by dividing the values row-wise with the sum. To counteract zero division
        all nan values are set to zero.
        Args:
            rule_matches: Binary encoded array of which rules matched. Shape: instances x rules
            mapping_rules_labels: Mapping of rules to labels, binary encoded. Shape: rules x classes

        Returns:

        """
        rule_counts = np.matmul(rule_matches, mapping_rules_labels)
        rule_counts_probs = rule_counts / rule_counts.sum(axis=1).reshape(-1, 1)

        rule_counts_probs[np.isnan(rule_counts_probs)] = 0
        return rule_counts_probs

    def test(self, test_features: Tensor, test_labels: Tensor):
        self.model.eval()

        predictions = self.model(test_features)

        acc = accuracy_of_probs(predictions, test_labels)
        logger.info("Accuracy is {}".format(acc.detach()))
        return acc
