import logging

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import TensorDataset
from sklearn.metrics import classification_report

from knodle.transformation.majority import input_to_majority_vote_input
from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.trainer import Trainer
from knodle.trainer.config import MajorityConfig
from knodle.trainer.utils.utils import log_section, accuracy_of_probs

logger = logging.getLogger(__name__)


class NoDenoisingTrainer(Trainer):
    """
    The baseline class implements a baseline model for labeling data with weak supervision.
        A simple majority vote is used for this purpose.
    """

    def __init__(
            self,
            model: nn.Module,
            mapping_rules_labels_t: np.ndarray,
            model_input_x: TensorDataset,
            rule_matches_z: np.ndarray,
            dev_model_input_x: TensorDataset = None,
            dev_gold_labels_y: TensorDataset = None,
            trainer_config: MajorityConfig = None,
    ):
        if trainer_config is None:
            trainer_config = MajorityConfig(optimizer=SGD(model.parameters(), lr=0.001))
        super().__init__(
            model, mapping_rules_labels_t, model_input_x, rule_matches_z, trainer_config=trainer_config
        )

        self.dev_model_input_x = dev_model_input_x
        self.dev_gold_labels_y = dev_gold_labels_y

    def _load_batch(self, batch):

        input_batch = [inp.to(self.trainer_config.device) for inp in batch[0: -1]]
        label_batch = batch[-1].to(self.trainer_config.device)

        return input_batch, label_batch

    def train_loop(self, feature_label_dataloader):
        log_section("Training starts", logger)

        self.model.to(self.trainer_config.device)
        self.model.train()
        for current_epoch in tqdm(range(self.trainer_config.epochs)):
            epoch_loss, epoch_acc = 0.0, 0.0
            logger.info("Epoch: {}".format(current_epoch))
            i = 0
            for batch in tqdm(feature_label_dataloader):
                input_batch, label_batch = self._load_batch(batch)

                # forward pass
                self.trainer_config.optimizer.zero_grad()
                outputs = self.model(*input_batch)
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs[0]
                    
                loss = self.trainer_config.criterion(logits, label_batch)

                # backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.trainer_config.optimizer.step()
                acc = accuracy_of_probs(logits, label_batch)

                epoch_loss += loss.detach().item()
                epoch_acc += acc.item()

                # i += 1
                # if i > 0:
                #     break

            avg_loss = epoch_loss / len(feature_label_dataloader)
            avg_acc = epoch_acc / len(feature_label_dataloader)

            logger.info("Epoch train loss: {}".format(avg_loss))
            logger.info("Epoch train accuracy: {}".format(avg_acc))

            if self.dev_model_input_x is not None and self.dev_gold_labels_y is not None:
                clf_report = self.test(self.dev_model_input_x, self.dev_gold_labels_y)
                logger.info("Epoch development accuracy: {}".format(clf_report["accuracy"]))

        log_section("Training done", logger)

        self.model.eval()

    def train(self):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """
        model_input_x, label_probs = input_to_majority_vote_input(
            self.model_input_x, self.rule_matches_z, self.mapping_rules_labels_t,
            filter_non_labelled=self.trainer_config.filter_non_labelled,
            other_class_id=self.trainer_config.other_class_id
        )

        feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self.train_loop(feature_label_dataloader)

    def test(self, features_dataset: TensorDataset, labels: TensorDataset):

        feature_label_dataset = input_labels_to_tensordataset(features_dataset, labels.tensors[0].cpu().numpy())
        feature_label_dataloader = self._make_dataloader(feature_label_dataset, shuffle=False)

        self.model.to(self.trainer_config.device)
        self.model.eval()
        predictions_list, label_list = [], []
        i = 0
        # Loop over predictions
        with torch.no_grad():
            for batch in tqdm(feature_label_dataloader):
                input_batch, label_batch = self._load_batch(batch)

                # forward pass
                self.trainer_config.optimizer.zero_grad()
                outputs = self.model(*input_batch)
                if isinstance(outputs, torch.Tensor):
                    prediction_vals = outputs
                else:
                    prediction_vals = outputs[0]

                # add predictions and labels
                predictions = np.argmax(prediction_vals.detach().cpu().numpy(), axis=-1)
                predictions_list.append(predictions)
                label_list.append(label_batch.detach().cpu().numpy())

                # i += 1
                # if i > 0:
                #     break

        # transform to correct format.
        predictions = np.squeeze(np.hstack(predictions_list))
        gold_labels = np.squeeze(np.hstack(label_list))

        clf_report = classification_report(y_true=gold_labels, y_pred=predictions, output_dict=True)

        return clf_report
