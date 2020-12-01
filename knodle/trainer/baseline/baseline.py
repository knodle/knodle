from torch.nn import Module
from torch import Tensor
import numpy as np

from knodle.final_label_decider.FinalLabelDecider import get_majority_vote_probabilities
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
            inputs: Input tensors. These tensors will be fed to the provided model (instaces x features)
            rule_matches: All rule matches (instances x rules)
            epochs: Epochs to train
        """

        if epochs <= 0:
            raise ValueError("Epochs needs to be positive")

        self.model.train()
        labels = get_majority_vote_probabilities(
            rule_matches=rule_matches,
            output_classes=self.trainer_config.output_classes,
        )

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

    def denoise_rule_matches(self, rule_matches: np.ndarray) -> np.ndarray:
        """
        The baseline model trainer doesn't denoise the rule_matches. Therefore, the same array is returned
        Returns:

        """
        return rule_matches

    def test(self, test_features: Tensor, test_labels: Tensor):
        self.model.eval()

        predictions = self.model(test_features)

        acc = accuracy_of_probs(predictions, test_labels)
        logger.info("Accuracy is {}".format(acc.detach()))
        return acc
