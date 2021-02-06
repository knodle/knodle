import logging

import numpy as np
from torch.nn import Module
from torch.utils.data import TensorDataset
from tqdm import tqdm

from knodle.transformation.majority import input_to_majority_vote_input
from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.trainer import Trainer
from knodle.trainer.baseline.majority_config import MajorityConfig
from knodle.trainer.utils.utils import (
    log_section,
    accuracy_of_probs
)

logger = logging.getLogger(__name__)


class NoDenoisingTrainer(Trainer):
    """
    The baseline class implements a baseline model for labeling data with weak supervision.
        A simple majority vote is used for this purpose.
    """

    def __init__(
            self,
            model: Module,
            mapping_rules_labels_t: np.ndarray,
            model_input_x: TensorDataset,
            rule_matches_z: np.ndarray,
            trainer_config: MajorityConfig = None,
    ):
        if trainer_config is None:
            trainer_config = MajorityConfig(model=model)
        super().__init__(
            model, mapping_rules_labels_t, model_input_x, rule_matches_z, trainer_config
        )

    def train(self):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """

        model_input_x, label_probs = input_to_majority_vote_input(
            self.model_input_x, self.rule_matches_z, self.mapping_rules_labels_t,
            filter_empty_z_rows=self.trainer_config.filter_emtpy_z_rows
        )

        feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        log_section("Training starts", logger)
        device = self.trainer_config.device
        self.model.to(self.trainer_config.device)
        self.model.train()
        for current_epoch in tqdm(range(self.trainer_config.epochs)):
            epoch_loss, epoch_acc = 0.0, 0.0
            logger.info("Epoch: {}".format(current_epoch))

            for feature_batch, label_batch in feature_label_dataloader:
                feature_batch = feature_batch.to(device)
                label_batch = label_batch.to(device)

                self.model.zero_grad()
                predictions = self.model(feature_batch)
                loss = self.trainer_config.criterion(predictions, label_batch)
                loss.backward()
                self.trainer_config.optimizer.step()
                acc = accuracy_of_probs(predictions, label_batch)

                epoch_loss += loss.detach()
                epoch_acc += acc.item()

            avg_loss = epoch_loss / len(feature_label_dataloader)
            avg_acc = epoch_acc / len(feature_label_dataloader)

            logger.info("Epoch loss: {}".format(avg_loss))
            logger.info("Epoch Accuracy: {}".format(avg_acc))

        log_section("Training done", logger)
