from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset

from knodle.trainer.config.trainer_config import TrainerConfig
import logging
import numpy as np

from knodle.trainer.utils.utils import accuracy_of_probs

logger = logging.getLogger(__name__)


class DsModelTrainer(ABC):
    def __init__(
        self,
        model: Module,
        mapping_rules_labels_t: np.ndarray,
        model_input_x: TensorDataset,
        rule_matches_z: np.ndarray,
        trainer_config: TrainerConfig = None,
    ):
        """
        Constructor for each DsModelTrainer.
            Args:
                model: PyTorch model which will be used for final classification.
                mapping_rules_labels_t: Mapping of rules to labels, binary encoded. Shape: rules x classes
                model_input_x: Input tensors. These tensors will be fed to the provided model.
                rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
                trainer_config: Config for different parameters like loss function, optimizer, batch size.
        """
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.mapping_rules_labels_t = mapping_rules_labels_t
        self.model_input_x = model_input_x
        self.rule_matches_z = rule_matches_z

        if trainer_config is None:
            self.trainer_config = TrainerConfig(self.model)
            self.logger.info(
                "Default trainer Config is used: {}".format(self.trainer_config)
            )
        else:
            self.trainer_config = trainer_config
            self.logger.info(
                "Initalized trainer with custom trainer config: {}".format(
                    self.trainer_config.__dict__
                )
            )

    @abstractmethod
    def train(self):
        pass

    def test(self, test_features: Tensor, test_labels: Tensor):
        self.model.eval()

        predictions = self.model(test_features)

        acc = accuracy_of_probs(predictions, test_labels)
        logger.info("Accuracy is {}".format(acc.detach()))
        return acc
