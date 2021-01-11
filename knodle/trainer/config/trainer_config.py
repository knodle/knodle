from typing import Callable
from snorkel.classification import cross_entropy_with_probs
from torch import Tensor
from torch.nn import Module
from torch.optim import SGD, optimizer


class TrainerConfig:
    def __init__(
        self,
        model: Module,
        criterion: Callable[[Tensor, Tensor], float] = cross_entropy_with_probs,
        batch_size: int = 32,
        optimizer_: optimizer = None,
        output_classes: int = 2,
        lr: float = 0.01,
        epochs: int = 3,
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
