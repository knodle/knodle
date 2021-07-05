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
        self.trainer = []
        configs = copy.deepcopy(kwargs["trainer_config"])
        for i, n in enumerate(name):
            kwargs["trainer_config"] = configs[i]
            self.trainer.append(AutoTrainer(n, **kwargs))

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ):
        for i in range(len(self.trainer)):
            trainer_name = str(type(self.trainer[i].trainer)).split(".")[-1][:-2]
            log_section(f"Training {trainer_name}", logger)
            self.trainer[i].train()
        # all_trainers = copy.deepcopy(self.trainer)
        # trained_trainers = []
        # for trainer in all_trainers:
        #     log_section(f"Training trainer {trainer}", logger)
        #     self.trainer = copy.deepcopy(trainer)
        #     trained_trainers.append(self.trainer.train(
        #         model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y
        #     ))
        # self.trainer = trained_trainers
        # return trained_trainers

    def test(self, test_features: TensorDataset, test_labels: TensorDataset) -> Dict:
        metrics = {}
        for i in range(len(self.trainer)):
            trainer_name = str(type(self.trainer[i].trainer)).split(".")[-1][:-2]
            metrics[trainer_name] = self.trainer[i].test(test_features, test_labels)
        return metrics
        # all_trainers_metrics = {}
        # all_trainers = copy.deepcopy(self.trainer)
        # for trainer in all_trainers:
        #     log_section(f"Testing trainer {trainer}", logger)
        #     self.trainer = copy.deepcopy(trainer)
        #     trainer_name = str(type(trainer)).split(".")[-1][:-2]
        #     all_trainers_metrics[trainer_name] = self.trainer.test(test_features, test_labels)
        # return all_trainers_metrics
