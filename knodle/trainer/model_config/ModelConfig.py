from typing import Callable
from snorkel.classification import cross_entropy_with_probs
from torch import Tensor
from torch.nn import Module
from torch.optim import SGD


class ModelConfig:
    def __init__(
        self,
        model: Module,
        criterion: Callable[[Tensor, Tensor], float] = cross_entropy_with_probs,
        learning_rate: float = 0.1,
        epsilon: float = 0.01,
        batch_size: int = 32,
        optimizer=SGD,
        output_classes: int = 2,
    ):
        self.criterion = criterion
        self.batch_size = batch_size
        self.optimizer = optimizer(model.parameters(), learning_rate, epsilon)
        self.output_classes = output_classes
