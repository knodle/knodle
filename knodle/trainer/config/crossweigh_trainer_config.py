from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.optim import SGD, optimizer

DEFAULT_LEARNING_RATE = 0.01


class TrainerConfig:
    def __init__(self,
                 model: Module,
                 criterion: Callable[[Tensor, Tensor], float] = None,
                 batch_size: int = 32,
                 optimizer_: optimizer = None,
                 output_classes: int = 2,
                 epochs: int = 2,
                 class_weights: Tensor = None,
                 seed: int = 12345,      # set seed for reproducibility
                 enable_cuda: bool = False,
                 use_grad_clipping: bool = True,
                 grad_clipping: int = 5
                 ):
        self.criterion = criterion
        self.batch_size = batch_size
        self.seed = seed
        self.enable_cuda = enable_cuda
        self.use_grad_clipping = use_grad_clipping
        self.grad_clipping = grad_clipping

        if epochs <= 0:
            raise ValueError("Epochs needs to be positive")
        self.epochs = epochs

        if class_weights is None:
            self.class_weights = torch.tensor([1.0] * self.output_classes)
        else:
            if len(class_weights) != output_classes:
                raise Exception("Wrong class weights initialisation!")
            self.class_weights = class_weights

        if optimizer_ is None:
            self.optimizer = SGD(model.parameters(), lr=DEFAULT_LEARNING_RATE)
        else:
            self.optimizer = optimizer_
        self.output_classes = output_classes

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')
        else:
            self.criterion = criterion
