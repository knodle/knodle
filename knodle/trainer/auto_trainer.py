from typing import Callable

import numpy as np
from torch.utils.data import TensorDataset

from knodle.trainer.trainer import Trainer


class AutoTrainer:
    """ The factory class for creating training executors
    See See https://medium.com/@geoffreykoh/implementing-the-factory-
    pattern-via-dynamic-registry-and-python-decorators-479fc1537bbe
    """

    registry = {}
    """ Internal registry for available trainers """

    def __init__(self, name, **kwargs):
        self.trainer = self.create_trainer(name, **kwargs)

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ):
        return self.trainer.train(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)

    def test(self, test_features: TensorDataset, test_labels: TensorDataset):
        return self.trainer.test(test_features, test_labels)

    @classmethod
    def create_trainer(cls, name: str, **kwargs) -> Trainer:
        """ Factory command to create the executor """

        exec_class = cls.registry[name]
        executor = exec_class(**kwargs)
        return executor

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Trainer) -> Callable:
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper
