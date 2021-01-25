import logging

import numpy as np
from tqdm.auto import tqdm

import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from sklearn.metrics import classification_report

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
            TensorDataset(self.model_input_x.tensors[0], self.model_input_x.tensors[1], torch.from_numpy(labels))
        )

        log_section("Training starts", logger)
        self.model.train()

        for current_epoch in range(self.trainer_config.epochs):
            epoch_loss, epoch_acc = 0.0, 0.0
            logger.info("Epoch: {}".format(current_epoch))
            i = 0
            for input_ids_batch, attention_mask_batch, label_batch in tqdm(feature_label_dataloader):
                i = i + 1
                inputs = {
                    "input_ids": input_ids_batch.to(device),
                    "attention_mask": attention_mask_batch.to(device),
                }
                labels = label_batch.to(device)

                # forward pass
                self.trainer_config.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = self.trainer_config.criterion(outputs[0], labels)

                # backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.trainer_config.optimizer.step()
                acc = accuracy_of_probs(outputs[0], labels)

                epoch_loss += loss.detach()
                epoch_acc += acc.item()
                if i > 0:
                    break

            avg_loss = epoch_loss / len(feature_label_dataloader)
            avg_acc = epoch_acc / len(feature_label_dataloader)

            logger.info("Epoch loss: {}".format(avg_loss))
            logger.info("Epoch Accuracy: {}".format(avg_acc))

        del loss
        log_section("Training done", logger)

        self.model.eval()

    def test(self, features_dataset: TensorDataset, labels: TensorDataset):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)

        feature_labels_dataloader = self._make_dataloader(
            TensorDataset(
                features_dataset.model_input_x.tensors[0], features_dataset.model_input_x.tensors[1],
                labels.tensors[0]
            )
        )

        self.model.eval()
        with torch.no_grad():
            all_predictions, all_labels = torch.Tensor(), torch.Tensor()
            for input_ids_batch, attention_mask_batch, label_batch in feature_labels_dataloader:
                inputs = {
                    "input_ids": input_ids_batch.to(device),
                    "attention_mask": attention_mask_batch.to(device),
                }
                labels = label_batch.to(device)

                # forward pass
                self.trainer_config.optimizer.zero_grad()
                outputs = self.model(**inputs)
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
