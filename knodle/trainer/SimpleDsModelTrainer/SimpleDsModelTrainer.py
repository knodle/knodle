from torch.nn import Module
from torch import Tensor
import numpy as np

from knodle.final_label_decider.FinalLabelDecider import get_majority_vote_probabilities
from knodle.model import LogisticRegressionModel
import logging

from knodle.trainer.config.TrainerConfig import TrainerConfig

logger = logging.getLogger(__name__)


class SimpleDsModelTrainer:
    def __init__(self, model: Module, trainer_config: TrainerConfig = None):
        """
            The SimpleDsModelTrainer class implements a baseline model for labeling data with weak supervision.
        A simple majority vote is used for this purpose.
            Args:
                model: PyTorch model which will be used for final classification.
                trainer_config: Config for different parameters like loss function, optimizer, batch size.
        """
        self.model = model
        if trainer_config is None:
            self.trainer_config = TrainerConfig(self.model)
            logger.info("Default Model Config is used: {}".format(self.trainer_config))
        else:
            self.trainer_config = trainer_config
            logger.info(
                "Initalized trainer with custom model config: {}".format(
                    self.trainer_config.__dict__
                )
            )

    def train(self, inputs: Tensor, rule_matches: np.ndarray, epochs: int):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        Args:
            inputs: Input tensors. These tensors will be fed to the provided model (instaces x features)
            rule_matches: All rule matches (instances x rules)
            epochs: Epochs to train
        """
        self.model.train()
        labels = get_majority_vote_probabilities(
            rule_matches=rule_matches,
            output_classes=self.trainer_config.output_classes,
        )

        labels = Tensor(labels)

        for current_epoch in range(epochs):
            logger.info("Epoch: ", current_epoch)
            self.model.zero_grad()
            predictions = self.model(inputs)
            loss = self.trainer_config.criterion(predictions, labels)
            logger.info("Loss is: ", loss.float())
            loss.backward()
            self.trainer_config.optimizer.step()


if __name__ == "__main__":
    logistic_regression = LogisticRegressionModel(10, 2)
    obj = SimpleDsModelTrainer(logistic_regression)
