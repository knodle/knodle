import torch
from torch import Tensor

from knodle.trainer.auto_config import AutoConfig
from knodle.trainer.config import BaseTrainerConfig


@AutoConfig.register("majority")
class MajorityConfig(BaseTrainerConfig):
    def __init__(
            self,
            use_probabilistic_labels: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.use_probabilistic_labels = use_probabilistic_labels
