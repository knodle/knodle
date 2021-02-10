from typing import Callable

from snorkel.classification import cross_entropy_with_probs
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import SGD, optimizer

from knodle.trainer.utils.utils import check_and_return_device


class TrainerConfig:
    def __init__(
            self,
            model: Module,
            criterion: Callable[[Tensor, Tensor], float] = cross_entropy_with_probs,
            batch_size: int = 32,
            optimizer_: optimizer = None,
            output_classes: int = 2,
            lr: float = 0.01,
            epochs: int = 35,
            seed: int = 42
    ):
        self.criterion = criterion
        self.batch_size = batch_size

        if epochs <= 0:
            raise ValueError("Epochs needs to be positive")
        self.epochs = epochs

        if optimizer_ is None:
            self.optimizer = SGD(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer_
        self.output_classes = output_classes
        self.device = check_and_return_device()
        self.seed = seed
        torch.manual_seed(self.seed)


class MajorityConfig(TrainerConfig):
    def __init__(
            self,
            filter_non_labelled: bool = True,
            use_probabilistic_labels: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.filter_non_labelled = filter_non_labelled
        self.use_probabilistic_labels = use_probabilistic_labels
