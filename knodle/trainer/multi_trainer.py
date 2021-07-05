import copy
import logging
from typing import Callable, List, Dict

import numpy as np
from torch.utils.data import TensorDataset

from knodle.trainer import AutoTrainer
from knodle.trainer.utils import log_section


logger = logging.getLogger(__name__)


class MultiTrainer:
    """ The factory class for creating training executors
    See See https://medium.com/@geoffreykoh/implementing-the-factory-
    pattern-via-dynamic-registry-and-python-decorators-479fc1537bbe
    """

    registry = {}
    """ Internal registry for available trainers """

    def __init__(self, name: List, **kwargs):
        self.trainer, configs = [], []
        if "trainer_config" in kwargs:
            configs = copy.deepcopy(kwargs["trainer_config"])
        for i, n in enumerate(name):
            if configs:
                kwargs["trainer_config"] = configs[i]
            self.trainer.append(copy.deepcopy(AutoTrainer(n, **kwargs).trainer))

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ):
        for i in range(len(self.trainer)):
            trainer_name = str(type(self.trainer[i])).split(".")[-1][:-2]
            log_section(f"Training {trainer_name}", logger)
            self.trainer[i].train()

    def test(self, test_features: TensorDataset, test_labels: TensorDataset) -> Dict:
        metrics = {}
        for i in range(len(self.trainer)):
            trainer_name = str(type(self.trainer[i])).split(".")[-1][:-2]
            metrics[trainer_name] = self.trainer[i].test(test_features, test_labels)
        return metrics
