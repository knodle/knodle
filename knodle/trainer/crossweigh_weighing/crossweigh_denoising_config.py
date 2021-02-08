from typing import Callable

import torch
import torch.nn as nn
from snorkel.classification import cross_entropy_with_probs
from torch import Tensor
from torch.optim import optimizer
from knodle.trainer.utils.utils import check_and_return_device


class CrossWeighDenoisingConfig:
    def __init__(
            self,
            model: nn.Module,
            crossweigh_partitions: int = 3,
            crossweigh_folds: int = 5,
            crossweigh_epochs: int = 2,
            weight_reducing_rate: float = 0.5,
            samples_start_weights: float = 3.0,
            no_match_weights: float = 0.5,
            size_factor: int = 200,
            batch_size: int = 32,
            lr: float = 0.01,
            output_classes: int = 2,
            class_weights: Tensor = None,
            optimizer_: optimizer = None,
            criterion: Callable[[Tensor, Tensor], float] = cross_entropy_with_probs,
            seed: int = "12345",
            use_grad_clipping: bool = True,
            grad_clipping: int = 5,
            filter_empty_probs: bool = False,
            no_match_class_label: int = None):

        self.cw_partitions = crossweigh_partitions
        self.cw_folds = crossweigh_folds
        self.cw_epochs = crossweigh_epochs
        self.weight_reducing_rate = weight_reducing_rate
        self.samples_start_weights = samples_start_weights
        self.no_match_weights = no_match_weights
        self.size_factor = size_factor
        self.batch_size = batch_size
        self.lr = lr
        self.output_classes = output_classes
        self.criterion = criterion
        self.seed = seed
        self.use_grad_clipping = use_grad_clipping
        self.grad_clipping = grad_clipping
        self.filter_empty_probs = filter_empty_probs
        self.no_match_class_label = no_match_class_label
        self.device = check_and_return_device()

        self.criterion = criterion

        if class_weights is None:
            self.class_weights = torch.tensor([1.0] * self.output_classes)
        else:
            if len(class_weights) != output_classes:
                raise Exception("Wrong class sample_weights initialisation!")
            self.class_weights = class_weights

        if optimizer_ is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        else:
            self.optimizer = optimizer_
