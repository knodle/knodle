from typing import Callable

from snorkel.classification import cross_entropy_with_probs
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from knodle.trainer.utils.utils import check_and_return_device, set_seed


class TrainerConfig:
    def __init__(
            self,
            criterion: Callable[[Tensor, Tensor], float] = cross_entropy_with_probs,
            batch_size: int = 32,
            optimizer: Optimizer = None,
            output_classes: int = 2,
            epochs: int = 35,
            seed: int = 42,
            filter_non_labelled: bool = True,
            use_probabilistic_labels: bool = True,
            other_class_id: int = None,
            grad_clipping: int = None,
            class_weights: Tensor = None,
    ):
        self.criterion = criterion
        self.batch_size = batch_size
        set_seed(seed)

        if epochs <= 0:
            raise ValueError("Epochs needs to be positive")
        self.epochs = epochs

        if optimizer is None:
            raise ValueError("An optimizer needs to be provided")
        else:
            self.optimizer = optimizer
        self.output_classes = output_classes
        self.device = check_and_return_device()

        self.filter_non_labelled = filter_non_labelled
        self.use_probabilistic_labels = use_probabilistic_labels
        self.other_class_id = other_class_id
        self.grad_clipping = grad_clipping

        if class_weights is None:
            self.class_weights = class_weights
        else:
            if len(class_weights) != self.output_classes:
                raise Exception("Wrong class sample_weights initialisation!")
            self.class_weights = class_weights