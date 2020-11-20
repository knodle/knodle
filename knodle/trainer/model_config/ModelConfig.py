from typing import Callable
from snorkel.classification import cross_entropy_with_probs
from torch import Tensor
from torch.nn import Module
from torch.optim import SGD, optimizer

DEFAULT_LEARNING_RATE = 0.01


class ModelConfig:
    def __init__(
        self,
        model: Module,
        criterion: Callable[[Tensor, Tensor], float] = cross_entropy_with_probs,
        batch_size: int = 32,
        optimizer_: optimizer = None,
        output_classes: int = 2,
    ):
        self.criterion = criterion
        self.batch_size = batch_size

        if optimizer_ is None:
            # Set default
            self.optimizer = SGD(model.parameters(), lr=DEFAULT_LEARNING_RATE)
        else:
            self.optimizer = optimizer_
        self.output_classes = output_classes
