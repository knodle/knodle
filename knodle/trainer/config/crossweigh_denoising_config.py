from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import optimizer

LEARNING_RATE = 0.01


class CrossWeighDenoisingConfig:
    def __init__(self,
                 model: nn.Module,
                 crossweigh_partitions: int = 2,
                 crossweigh_folds: int = 3,
                 crossweigh_epochs: int = 1,
                 weight_reducing_rate: int = 0.7,
                 samples_start_weights: int = 4.0,
                 no_relation_weights: int = 0.5,
                 size_factor: int = 200,
                 batch_size: int = 64,
                 output_classes: int = 0,
                 class_weights: Tensor = None,
                 optimizer_: optimizer = None,
                 criterion: Callable[[Tensor, Tensor], float] = None,
                 path_to_weights: str = "data/weights",
                 seed: int = "12345",
                 enable_cuda: bool = False,
                 use_grad_clipping: bool = True,
                 grad_clipping: int = 5,
                 ):

        self.cw_partitions = crossweigh_partitions
        self.cw_folds = crossweigh_folds
        self.cw_epochs = crossweigh_epochs
        self.weight_reducing_rate = weight_reducing_rate
        self.samples_start_weights = samples_start_weights
        self.no_relation_weights = no_relation_weights
        self.size_factor = size_factor
        self.batch_size = batch_size
        self.output_classes = output_classes
        self.criterion = criterion
        self.path_to_weights = path_to_weights
        self.seed = seed
        self.enable_cuda = enable_cuda
        self.use_grad_clipping = use_grad_clipping
        self.grad_clipping = grad_clipping

        if class_weights is None:
            self.class_weights = torch.tensor([1.0] * self.output_classes)
        else:
            if len(class_weights) != output_classes:
                raise Exception("Wrong class weights initialisation!")
            self.class_weights = class_weights

        if optimizer_ is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        else:
            self.optimizer = optimizer_

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = criterion
