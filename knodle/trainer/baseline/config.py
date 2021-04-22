import torch
from torch import Tensor
from typing import Callable

from knodle.trainer.auto_config import AutoConfig
from knodle.trainer.config import BaseTrainerConfig
from knodle.utils.losses.cross_entropy import cross_entropy


@AutoConfig.register("majority")
class MajorityConfig(BaseTrainerConfig):
    def __init__(
            self,
            use_probabilistic_labels: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.use_probabilistic_labels = use_probabilistic_labels
        if not use_probabilistic_labels:
            self.criterion: Callable[[Tensor, Tensor], float] = cross_entropy
