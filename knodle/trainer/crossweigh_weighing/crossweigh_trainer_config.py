from typing import Callable
import torch
import torch.nn as nn
from snorkel.classification import cross_entropy_with_probs
from torch import Tensor
from torch.nn import Module
from torch.optim import SGD, optimizer


class CrossWeighTrainerConfig:
    def __init__(
            self,
            model: Module,
            criterion: Callable[[Tensor, Tensor], float] = cross_entropy_with_probs,
            batch_size: int = 32,
            optimizer_: optimizer = None,
            output_classes: int = 2,
            lr: float = 0.01,
            epochs: int = 2,
            class_weights: Tensor = None,
            seed: int = 12345,  # set seed for reproducibility
            enable_cuda: bool = False,
            use_grad_clipping: bool = True,
            grad_clipping: int = 5,
            no_match_class_label: int = None
    ):
        self.criterion = criterion
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.enable_cuda = enable_cuda
        self.use_grad_clipping = use_grad_clipping
        self.grad_clipping = grad_clipping
        self.output_classes = output_classes
        self.no_match_class_label = no_match_class_label

        if epochs <= 0:
            raise ValueError("Epochs needs to be positive")
        self.epochs = epochs

        self.criterion = criterion

        if class_weights is None:
            self.class_weights = torch.tensor([1.0] * self.output_classes)
        else:
            if len(class_weights) != output_classes:
                raise Exception("Wrong class sample_weights initialisation!")
            self.class_weights = class_weights

        if optimizer_ is None:
            self.optimizer = SGD(model.parameters(), lr=self.lr)
        else:
            self.optimizer = optimizer_
