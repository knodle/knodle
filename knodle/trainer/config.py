import pathlib
from typing import Callable, Dict
import os
import logging

from snorkel.classification import cross_entropy_with_probs

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from knodle.trainer.utils.utils import check_and_return_device, set_seed

logger = logging.getLogger(__name__)


class TrainerConfig:
    def __init__(
            self,
            criterion: Callable[[Tensor, Tensor], float] = cross_entropy_with_probs,
            batch_size: int = 32,
            optimizer: Optimizer = None,
            lr: int = 0.01,
            output_classes: int = 2,
            class_weights: Tensor = None,
            epochs: int = 3,
            seed: int = 42,
            grad_clipping: int = None,
            device: str = None,
            caching_folder: str = None,
            caching_suffix: str = "",
            saved_models_dir: str = None
    ):
        self.seed = seed
        set_seed(seed)

        # create directory where saved models will be stored
        self.saved_models_dir = saved_models_dir
        if self.saved_models_dir is not None:
            os.makedirs(self.saved_models_dir, exist_ok=True)

        self.criterion = criterion
        self.lr = lr
        self.batch_size = batch_size
        self.output_classes = output_classes
        self.grad_clipping = grad_clipping
        self.device = torch.device(device) if device is not None else check_and_return_device()
        logger.info(f"Model will be trained on {self.device}")

        self.caching_suffix = caching_suffix
        if caching_folder is not None:
            self.caching_folder = caching_folder
        else:
            self.caching_folder = os.path.join(pathlib.Path().absolute(), "cache")
        logger.info(f"The cache will be saved to {self.caching_folder} folder")

        if epochs <= 0:
            raise ValueError("Epochs needs to be positive")
        self.epochs = epochs

        if optimizer is None:
            raise ValueError("An optimizer needs to be provided")
        else:
            self.optimizer = optimizer

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
            evaluate_with_other_class: bool = False,
            ids2labels: Dict = None,
            **kwargs
    ):
        """ Additionally provided parameters needed for handling the cases where there are data samples with no rule
         matched (filtering OR introducing the other class + training&evaluation with other class) """
        super().__init__(**kwargs)
        self.filter_non_labelled = filter_non_labelled
        self.other_class_id = other_class_id
        self.evaluate_with_other_class = evaluate_with_other_class
        self.ids2labels = ids2labels

        if self.other_class_id is not None and self.filter_non_labelled:
            raise ValueError("You can either filter samples with no weak labels or add them to 'other_class_id'")

        logger.debug(f"{self.evaluate_with_other_class} and {self.ids2labels}")
        if self.evaluate_with_other_class and self.ids2labels is None:
            # check if the selected evaluation type is valid
            logging.warning(
                "Labels to label ids correspondence is needed to make other_class specific evaluation. Since it is "
                "absent now, the standard sklearn metrics will be calculated instead."
            )
            self.evaluate_with_other_class = False
