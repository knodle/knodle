import pathlib
from typing import Callable, Dict
import os
import logging

from snorkel.classification import cross_entropy_with_probs

import torch
from torch import Tensor
from torch.optim import SGD
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
            seed: int = None,
            grad_clipping: int = None,
            device: str = None,
            caching_folder: str = os.path.join(pathlib.Path().absolute(), "cache"),
            caching_suffix: str = "",
            saved_models_dir: str = None
    ):
        """
        A default and minimum sufficient configuration of a Trainer instance.

        :param criterion: a usual PyTorch criterion; computes a gradient according to a given loss function
        :param batch_size: a usual PyTorch batch_size; the number of training examples utilized in one training iteration
        :param optimizer: a usual PyTorch optimizer; which is used to solve optimization problems by minimizing the
        function
        :param lr: a usual PyTorch learning rate; tuning parameter in an optimization algorithm that determines the step
        size at each iteration while moving toward a minimum of a loss function
        :param output_classes: the number of classes used in classification
        :param class_weights: introduce the weight of each class. By default, all classes have the same weights 1.0.
        :param epochs: the number of epochs the classification model will be trained
        :param seed: the desired seed for generating random numbers
        :param grad_clipping: if set to True, gradient norm of an iterable of parameters will be clipped
        :param device: what device the model will be trained on (CPU/CUDA)
        :param caching_folder: a path to the folder where cache will be saved (default: root/cache)
        :param caching_suffix: a specific index that could be added to the caching file name (e.g. in WSCrossWeigh for
        sample weights calculated in different iterations and stored in different files.)
        :param saved_models_dir: a path to the folder where trained models will be stored. If None, the trained models
        won't be stored.
        """
        self.seed = seed
        if self.seed is not None:
            set_seed(seed)

        self.caching_suffix = caching_suffix
        self.caching_folder = caching_folder
        os.makedirs(self.caching_folder, exist_ok=True)
        logger.info(f"The cache will be saved to {self.caching_folder} folder")

        # create directory where saved models will be stored
        if saved_models_dir:
            self.saved_models_dir = saved_models_dir
            os.makedirs(self.saved_models_dir, exist_ok=True)
        else:
            self.saved_models_dir = caching_folder
        logger.info(f"The trained models will be saved to the {self.saved_models_dir} directory.")

        self.criterion = criterion
        self.lr = lr
        self.batch_size = batch_size
        self.output_classes = output_classes
        self.grad_clipping = grad_clipping
        self.device = torch.device(device) if device is not None else check_and_return_device()
        logger.info(f"Model will be trained on {self.device}")

        if epochs <= 0:
            raise ValueError("Epochs needs to be positive")
        self.epochs = epochs

        if optimizer is None:
            logger.info(f"Defaulting to SGD optimizer as none specified in the config.")
            self.optimizer = SGD
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
            max_rules: int = None,
            min_coverage: float = None,
            drop_rules: bool = False,
            **kwargs
    ):
        """
        Additionally provided parameters needed for handling the cases where there are data samples with no rule
        matched (filtering OR introducing the other class + training & evaluation with other class).

        :param filter_non_labelled: if True, the samples with no rule matched will be filtered out from the dataset
        :param other_class_id: id of the negative class; if set, the samples with no rule matched will be assigned to it
        :param evaluate_with_other_class: if set to True, the evaluation will be done with respect to the negative class
        (for more details please see knodle/evaluation/other_class_metrics.py file)
        :param ids2labels: dictionary {label id: label}, which is needed to perform evaluation with the negative class
        """
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

        self.max_rules = max_rules
        self.min_coverage = min_coverage
        self.drop_rules = drop_rules
