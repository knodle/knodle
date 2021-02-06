import logging

import numpy as np
import torch
from abc import ABC, abstractmethod
from sklearn.metrics import classification_report
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
        self.mapping_rules_labels_t = mapping_rules_labels_t
        self.model_input_x = model_input_x
        self.rule_matches_z = rule_matches_z

        if trainer_config is None:
            self.trainer_config = TrainerConfig(model)

        else:
            self.trainer_config = trainer_config

        self.model = model.to(self.trainer_config.device)

    @abstractmethod
    def train(self):
        pass

    def test(self, test_features: TensorDataset, test_labels: TensorDataset):
        """
        Runs evaluation and returns a classification report with different evaluation metrics.
        Args:
            test_features: Test feature set
            test_labels: Gold label set.

        Returns:

        """
        predictions = self._prediction_loop(test_features, True)
        predictions, test_labels = (
            predictions.cpu().detach().numpy(),
            test_labels.tensors[0].cpu().detach().numpy(),
        )
        if predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)

        clf_report = classification_report(
            y_true=test_labels, y_pred=predictions, output_dict=True
        )
        logger.info("Accuracy is {}".format(clf_report["accuracy"]))
        return clf_report

    def _prediction_loop(self, features: TensorDataset, evaluate: bool):
        """
        This method returns all predictions of the model. Currently this function aims just for the test function.
        Args:
            features: DataSet with features to get predictions from
            evaluate: Boolean if model in evaluation mode or not.

        Returns:

        """
        feature_dataloader = self._make_dataloader(features)

        if evaluate:
            self.model.eval()
        else:
            self.model.train()

        predictions_list = torch.Tensor()

        for feature_counter, feature_batch in enumerate(feature_dataloader):
            # DISCUSS
            features = feature_batch[0].to(self.trainer_config.device)
            predictions = self.model(features)
            predictions_list = torch.cat([predictions_list, predictions.to("cpu")])

        return predictions_list

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
