import logging
import os

import pandas as pd
import numpy as np
from joblib import load, dump
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import TensorDataset

from knodle.trainer import TrainerConfig
from knodle.trainer.baseline.baseline import SimpleDsModelTrainer
from knodle.trainer.utils import log_section
from knodle.trainer.utils.denoise import get_majority_vote_probs
from knodle.trainer.utils.utils import accuracy_of_probs


logger = logging.getLogger(__name__)


def read_evaluation_data():
    imdb_dataset = pd.read_csv("../ImdbDataset/imdb_data_preprocessed.csv")
    rule_matches_z = load("../ImdbDataset/rule_matches.lib")
    mapping_rules_labels_t = load("../ImdbDataset/mapping_rules_labels.lib")
    return imdb_dataset, rule_matches_z, mapping_rules_labels_t


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


class MajorityBertTrainer(SimpleDsModelTrainer):
    def train(self):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)

        labels = get_majority_vote_probs(self.rule_matches_z, self.mapping_rules_labels_t)
        label_dataset = TensorDataset(Tensor(labels))

        feature_dataloader = self._make_dataloader(self.model_input_x)
        label_dataloader = self._make_dataloader(label_dataset)
        log_section("Training starts", logger)

        i = 0
        self.model.train()
        for current_epoch in tqdm(range(self.trainer_config.epochs)):
            epoch_loss, epoch_acc = 0.0, 0.0
            logger.info("Epoch: {}".format(current_epoch))

            for feature_batch, label_batch in zip(feature_dataloader, label_dataloader):

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

            avg_loss = epoch_loss / len(feature_dataloader)
            avg_acc = epoch_acc / len(feature_dataloader)

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
                # DISCUSS
                inputs = {
                    "input_ids": feature_batch[0],
                    "attention_mask": feature_batch[1],
                }
                predictions = self.model(**inputs)[0]
                predictions_list.append(predictions.detach().numpy())

        return torch.from_numpy(np.vstack(predictions_list))
