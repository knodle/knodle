import torch
from torch import Tensor

from knodle.trainer.auto_config import AutoConfig
from knodle.trainer.config import BaseTrainerConfig


@AutoConfig.register("majority")
class MajorityConfig(BaseTrainerConfig):
    def __init__(
            self,
            use_probabilistic_labels: bool = True,
            class_weights: Tensor = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.use_probabilistic_labels = use_probabilistic_labels

        if class_weights is None:
            self.class_weights = torch.tensor([1.0] * self.output_classes)
        else:
            if len(class_weights) != self.output_classes:
                raise Exception("Wrong class sample_weights initialisation!")
            self.class_weights = class_weights
