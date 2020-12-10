from abc import ABC
from torch.nn import Module

from knodle.trainer.config.TrainerConfig import TrainerConfig
import logging
import numpy as np


class DsModelTrainer(ABC):
    def __init__(
        self,
        model: Module,
        mapping_rules_labels_t: np.ndarray,
        trainer_config: TrainerConfig = None,
    ):
        """
        Constructor for each DsModelTrainer.
            Args:
                model: PyTorch model which will be used for final classification.
                trainer_config: Config for different parameters like loss function, optimizer, batch size.
                mapping_rules_labels_t: Mapping of rules to labels, binary encoded. Shape: rules x classes

        """
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.mapping_rules_labels_t = mapping_rules_labels_t

        if trainer_config is None:
            self.trainer_config = TrainerConfig(self.model)
            self.logger.info(
                "Default Model Config is used: {}".format(self.trainer_config)
            )
        else:
            self.trainer_config = trainer_config
            self.logger.info(
                "Initalized trainer with custom model config: {}".format(
                    self.trainer_config.__dict__
                )
            )
