import logging

import numpy as np
from torch import Tensor
from tqdm.auto import tqdm
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import TensorDataset
from sklearn.metrics import classification_report

from knodle.evaluation.plotting import draw_loss_accuracy_plot
from knodle.transformation.majority import input_to_majority_vote_input
from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.trainer import Trainer
from knodle.trainer.config import TrainerConfig
from knodle.trainer.utils.utils import log_section, accuracy_of_probs

logger = logging.getLogger(__name__)
PRINT_EVERY = 300


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
            trainer_config: TrainerConfig = None,
    ):
        if trainer_config is None:
            trainer_config = TrainerConfig(optimizer=SGD(model.parameters(), lr=0.001))
        super().__init__(
            model, mapping_rules_labels_t, model_input_x, rule_matches_z, trainer_config=trainer_config
        )

        self.dev_model_input_x = dev_model_input_x
        self.dev_gold_labels_y = dev_gold_labels_y

    def _load_batch(self, batch):

        input_batch = [inp.to(self.trainer_config.device) for inp in batch[0: -1]]
        label_batch = batch[-1].to(self.trainer_config.device)

        return input_batch, label_batch

    def train_loop(self, feature_label_dataloader, use_sample_weights: bool = False, draw_plot: bool = False):
        log_section("Training starts", logger)

        self.model.to(self.trainer_config.device)
        self.model.train()

        train_losses, train_acc = [], []
        if self.dev_model_input_x is not None:
            dev_losses, dev_acc = [], []

        for current_epoch in range(self.trainer_config.epochs):
            logger.info("Epoch: {}".format(current_epoch))
            epoch_loss, epoch_acc, steps = 0.0, 0.0, 0
            for batch in tqdm(feature_label_dataloader):
                input_batch, label_batch = self._load_batch(batch)
                steps += 1

                if use_sample_weights:
                    input_batch, sample_weights = input_batch[:-1], input_batch[-1]

                # forward pass
                self.trainer_config.optimizer.zero_grad()
                outputs = self.model(*input_batch)
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs[0]

                if use_sample_weights:
                    loss_no_reduction = self.trainer_config.criterion(
                        logits, label_batch, weight=self.trainer_config.class_weights, reduction="none"
                    )
                    loss = loss_no_reduction * sample_weights.mean()
                else:
                    loss = self.trainer_config.criterion(logits, label_batch, weight=self.trainer_config.class_weights)

                # backward pass
                loss.backward()
                if isinstance(self.trainer_config.grad_clipping, (int, float)):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.grad_clipping)
                self.trainer_config.optimizer.step()
                acc = accuracy_of_probs(logits, label_batch)

                epoch_loss += loss.detach().item()
                epoch_acc += acc.item()

                if steps % PRINT_EVERY == 0:
                    logger.info(f"Train loss: {epoch_loss / steps:.3f}, Train accuracy: {epoch_acc / steps:.3f}")

            avg_loss = epoch_loss / len(feature_label_dataloader)
            avg_acc = epoch_acc / len(feature_label_dataloader)
            train_losses.append(avg_loss)
            train_acc.append(avg_acc)

            logger.info("Epoch train loss: {}".format(avg_loss))
            logger.info("Epoch train accuracy: {}".format(avg_acc))

            if self.dev_model_input_x:
                dev_clf_report, dev_loss = self.test(self.dev_model_input_x, self.dev_gold_labels_y, loss_calculation=True)
                dev_losses.append(dev_loss)
                dev_acc.append(dev_clf_report["accuracy"])
                logger.info("Epoch development accuracy: {}".format(dev_clf_report["accuracy"]))

        log_section("Training done", logger)

        if draw_plot:
            if self.dev_model_input_x:
                draw_loss_accuracy_plot(
                    {"train loss": train_losses, "train acc": train_acc, "dev loss": dev_losses, "dev acc": dev_acc}
                )
            else:
                draw_loss_accuracy_plot({"train loss": train_losses, "train acc": train_acc})

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

    def test(self, features_dataset: TensorDataset, labels: TensorDataset, loss_calculation: bool = False):

        feature_label_dataset = input_labels_to_tensordataset(features_dataset, labels.tensors[0].cpu().numpy())
        feature_label_dataloader = self._make_dataloader(feature_label_dataset, shuffle=False)

        self.model.to(self.trainer_config.device)
        self.model.eval()
        predictions_list, label_list = [], []
        dev_loss, dev_acc = 0.0, 0.0

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

                if loss_calculation:
                    dev_loss += self._calculate_dev_loss(prediction_vals, label_batch.long())

                # add predictions and labels
                predictions = np.argmax(prediction_vals.detach().cpu().numpy(), axis=-1)
                predictions_list.append(predictions)
                label_list.append(label_batch.detach().cpu().numpy())

        # transform to correct format
        predictions = np.squeeze(np.hstack(predictions_list))
        gold_labels = np.squeeze(np.hstack(label_list))

        clf_report = classification_report(y_true=gold_labels, y_pred=predictions, output_dict=True)

        return clf_report, dev_loss / len(feature_label_dataloader)

    def _calculate_dev_loss(self, predictions: Tensor, labels: Tensor) -> Tensor:
        """ Calculates the loss on the dev set using given criterion"""
        predictions_one_hot = F.one_hot(predictions.argmax(1), num_classes=self.trainer_config.output_classes).float()
        labels_one_hot = F.one_hot(labels, self.trainer_config.output_classes)
        loss = self.trainer_config.criterion(predictions_one_hot, labels_one_hot)
        return loss.detach()
