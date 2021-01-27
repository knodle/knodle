import logging

import numpy as np
from torch import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset
from tqdm import tqdm

from knodle.trainer import TrainerConfig
from knodle.trainer.trainer import Trainer
from knodle.trainer.utils import log_section
from knodle.trainer.utils.denoise import get_majority_vote_probs
from knodle.trainer.utils.filter import filter_empty_probabilities
from knodle.trainer.utils.utils import (
    accuracy_of_probs,
    extract_tensor_from_dataset,
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
        trainer_config: TrainerConfig = None,
    ):
        super().__init__(
            model, mapping_rules_labels_t, model_input_x, rule_matches_z, trainer_config
        )

    def train(self):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """

        label_probs = get_majority_vote_probs(
            self.rule_matches_z, self.mapping_rules_labels_t
        )
        label_probs = Tensor(label_probs)
        label_probs = label_probs.to(self.trainer_config.device)

        model_input_x, label_probs = filter_empty_probabilities(self.model_input_x, label_probs)
        model_input_x_tensor = extract_tensor_from_dataset(model_input_x, 0)

        feature_label_dataset = TensorDataset(model_input_x_tensor, Tensor(label_probs))
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        log_section("Training starts", logger)

        self.model.train()
        for current_epoch in tqdm(range(self.trainer_config.epochs)):
            epoch_loss, epoch_acc = 0.0, 0.0
            logger.info("Epoch: {}".format(current_epoch))

            for feature_batch, label_batch in feature_label_dataloader:
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
