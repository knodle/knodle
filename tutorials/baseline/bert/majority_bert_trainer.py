import logging

import numpy as np
from tqdm.auto import tqdm

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from knodle.trainer.ds_model_trainer.ds_model_trainer import DsModelTrainer
from knodle.trainer.utils import log_section
from knodle.trainer.utils.denoise import get_majority_vote_probs
from knodle.trainer.utils.utils import accuracy_of_probs

logger = logging.getLogger(__name__)


class MajorityBertTrainer(DsModelTrainer):
    def train(self):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)

        labels = get_majority_vote_probs(self.rule_matches_z, self.mapping_rules_labels_t)

        feature_label_dataloader = self._make_dataloader(
            self.model_input_x.tensors[0], self.model_input_x.tensors[1], Tensor(labels)
        )

        log_section("Training starts", logger)
        self.model.train()

        for current_epoch in tqdm(range(self.trainer_config.epochs)):
            epoch_loss, epoch_acc = 0.0, 0.0
            logger.info("Epoch: {}".format(current_epoch))

            for feature_batch, label_batch in zip(feature_label_dataloader):
                inputs = {
                    "input_ids": feature_batch[0].to(device),
                    "attention_mask": feature_batch[1].to(device),
                }
                labels = label_batch[0].to(device)

                # forward pass
                self.trainer_config.optimizer.zero_grad()
                # TODO: possible device change
                outputs = self.model(**inputs)
                loss = self.trainer_config.criterion(outputs[0], labels)

                # backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.trainer_config.optimizer.step()
                # self.scheduler.step() # TODO
                acc = accuracy_of_probs(outputs[0], labels)

                epoch_loss += loss.detach()
                epoch_acc += acc.item()

            avg_loss = epoch_loss / len(feature_label_dataloader)
            avg_acc = epoch_acc / len(feature_label_dataloader)

            logger.info("Epoch loss: {}".format(avg_loss))
            logger.info("Epoch Accuracy: {}".format(avg_acc))

        del loss
        log_section("Training done", logger)

        self.model.eval()

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

        predictions_list = []
        with torch.no_grad():
            for feature_counter, feature_batch in enumerate(feature_dataloader):
                inputs = {
                    "input_ids": feature_batch[0],
                    "attention_mask": feature_batch[1],
                }
                predictions = self.model(**inputs)[0]
                predictions_list.append(predictions.detach().numpy())

        return torch.from_numpy(np.vstack(predictions_list))
