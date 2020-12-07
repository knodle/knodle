from abc import ABC, abstractmethod
import numpy as np
from torch import Tensor
from torch.nn import Module

from knodle.trainer.config.CrossWeightDenoisingConfig import CrossWeightDenoisingConfig
from knodle.trainer.config.TrainerConfig import TrainerConfig
import logging


class DsModelTrainer(ABC):
    def __init__(self,
                 model: Module,
                 trainer_config: TrainerConfig = None,
                 denoising_config: CrossWeightDenoisingConfig = None):
        """
        Constructor for each DsModelTrainer.
            Args:
                model: PyTorch model which will be used for final classification.
                trainer_config: Config for different parameters like loss function, optimizer, batch size.
        """
        self.model = model
        self.logger = logging.getLogger(__name__)

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

        if denoising_config is None:
            self.denoising_config = CrossWeightDenoisingConfig(self.model)
            self.logger.info(
                "Default denoising Config is used: {}".format(self.denoising_config)
            )
        else:
            self.denoising_config = denoising_config
            self.logger.info(
                "Initalized denoising with custom model config: {}".format(
                    self.denoising_config.__dict__
                )
            )

    @abstractmethod
    def train(self, inputs: Tensor, rule_matches: np.ndarray, epochs: int, **kwargs):
        pass

    @abstractmethod
    def denoise(self, rule_matches: np.ndarray, **kwargs) -> np.ndarray:
        pass
