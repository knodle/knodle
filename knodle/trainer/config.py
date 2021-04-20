from typing import Callable
import os

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
            class_weights: Tensor = None,
            epochs: int = 3,
            seed: int = 42,
            grad_clipping: int = None,
            device: str = None,
            output_dir_path: str = None
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
        set_seed(seed)
        self.grad_clipping = grad_clipping

        self.device = torch.device("device") if device is not None else check_and_return_device()
        # create model directory
        self.output_dir_path = output_dir_path
        if self.output_dir_path is not None:
            os.makedirs(self.output_dir_path, exist_ok=True)

        if class_weights is None:
            self.class_weights = torch.tensor([1.0] * self.output_classes)
        else:
            if len(class_weights) != self.output_classes:
                raise Exception("Wrong class sample_weights initialisation!")
            self.class_weights = class_weights


class BaseTrainerConfig(TrainerConfig):
    def __init__(
            self,
            filter_non_labelled: bool = True,
            other_class_id: int = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.filter_non_labelled = filter_non_labelled
        self.other_class_id = other_class_id


