import logging

import numpy as np
import torch
from abc import ABC, abstractmethod
from sklearn.metrics import classification_report
from torch import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader

from knodle.trainer.config import TrainerConfig

logger = logging.getLogger(__name__)


class Trainer(ABC):
    def __init__(
        self,
        model: Module,
        mapping_rules_labels_t: np.ndarray,
        model_input_x: TensorDataset,
        rule_matches_z: np.ndarray,
        trainer_config: TrainerConfig = None,
    ):
        """
        Constructor for each DsModelTrainer.
            Args:
                model: PyTorch model which will be used for final classification.
                mapping_rules_labels_t: Mapping of rules to labels, binary encoded. Shape: rules x classes
                model_input_x: Input tensors. These tensors will be fed to the provided model.
                rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
                trainer_config: Config for different parameters like loss function, optimizer, batch size.
        """
        self.model = model
        self.mapping_rules_labels_t = mapping_rules_labels_t
        self.model_input_x = model_input_x
        self.rule_matches_z = rule_matches_z

        if trainer_config is None:
            self.trainer_config = TrainerConfig(self.model)
            logger.info("Default trainer Config is used: {}".format(self.trainer_config))
        else:
            self.trainer_config = trainer_config
            logger.info("Initalized trainer with custom trainer config: {}".format(self.trainer_config.__dict__))

    @abstractmethod
    def train(self):
        pass

    def test(self, features_dataset: TensorDataset, labels: Tensor):
        feature_labels_dataset = TensorDataset(features_dataset.tensors[0], labels)
        feature_labels_dataloader = self._make_dataloader(feature_labels_dataset)

        self.model.eval()
        all_predictions, all_labels = torch.Tensor(), torch.Tensor()
        for features, labels in feature_labels_dataloader:
            outputs = self.model(features)
            _, predicted = torch.max(outputs, 1)
            all_predictions = torch.cat([all_predictions, predicted])
            all_labels = torch.cat([all_labels, labels])

        predictions, test_labels = (all_predictions.detach().numpy(), all_labels.detach().numpy())
        clf_report = classification_report(y_true=test_labels, y_pred=predictions, output_dict=True)

        logger.info(clf_report)
        logger.info("Accuracy: {}, ".format(clf_report["accuracy"]))
        print(clf_report)
        print("Accuracy: {}, ".format(clf_report["accuracy"]))
        return clf_report

    def _make_dataloader(
        self, dataset: TensorDataset, shuffle: bool = True
    ) -> DataLoader:
        dataloader = DataLoader(
            dataset,
            batch_size=self.trainer_config.batch_size,
            drop_last=False,
            shuffle=shuffle,
        )
        return dataloader
