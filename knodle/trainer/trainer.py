import os
import logging
from typing import Union, Dict, Tuple

import skorch
from tqdm.auto import tqdm
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import classification_report

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from knodle.evaluation.other_class_metrics import classification_report_other_class
from knodle.transformation.torch_input import input_labels_to_tensordataset, dataset_to_numpy_input
from knodle.transformation.rule_reduction import reduce_rule_matches
from knodle.evaluation.plotting import draw_loss_accuracy_plot

from knodle.trainer.config import TrainerConfig, BaseTrainerConfig
from knodle.trainer.utils.utils import log_section, accuracy_of_probs
from knodle.trainer.utils.checks import check_other_class_id

logger = logging.getLogger(__name__)


class Trainer(ABC):
    def __init__(
            self,
            model: Module,
            mapping_rules_labels_t: np.ndarray,
            model_input_x: TensorDataset,
            rule_matches_z: np.ndarray,
            dev_model_input_x: TensorDataset = None,
            dev_gold_labels_y: TensorDataset = None,
            trainer_config: TrainerConfig = None,
    ):
        """
        Constructor for each Trainer.
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
        self.dev_model_input_x = dev_model_input_x
        self.dev_gold_labels_y = dev_gold_labels_y

        if trainer_config is None:
            self.trainer_config = TrainerConfig(model)
        else:
            self.trainer_config = trainer_config

    @abstractmethod
    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ):
        pass

    @abstractmethod
    def test(self, test_features: TensorDataset, test_labels: TensorDataset):
        pass

    def initialise_optimizer(self):
        try:
            return self.trainer_config.optimizer(params=self.model.parameters(), lr=self.trainer_config.lr)
        except TypeError:
            logger.info("Wrong optimizer parameters. Optimizer should belong to torch.optim class or be PyTorch "
                        "compatible.")


class BaseTrainer(Trainer):

    def __init__(
            self,
            model: Module,
            mapping_rules_labels_t: np.ndarray,
            model_input_x: TensorDataset,
            rule_matches_z: np.ndarray,
            **kwargs):
        if kwargs.get("trainer_config", None) is None:
            kwargs["trainer_config"] = BaseTrainerConfig()
        super().__init__(model, mapping_rules_labels_t, model_input_x, rule_matches_z, **kwargs)

        check_other_class_id(self.trainer_config, self.mapping_rules_labels_t)

    def _load_train_params(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ):
        if model_input_x is not None and rule_matches_z is not None:
            self.model_input_x = model_input_x
            self.rule_matches_z = rule_matches_z
        if dev_model_input_x is not None and dev_gold_labels_y is not None:
            self.dev_model_input_x = dev_model_input_x
            self.dev_gold_labels_y = dev_gold_labels_y

    def _apply_rule_reduction(self):
        reduced_dict = reduce_rule_matches(
            rule_matches_z=self.rule_matches_z, mapping_rules_labels_t=self.mapping_rules_labels_t,
            drop_rules=self.trainer_config.drop_rules, max_rules=self.trainer_config.max_rules,
            min_coverage=self.trainer_config.min_coverage)
        self.rule_matches_z = reduced_dict["train_rule_matches_z"]
        self.mapping_rules_labels_t = reduced_dict["mapping_rules_labels_t"]

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

    def _load_batch(self, batch):

        input_batch = [inp.to(self.trainer_config.device) for inp in batch[0: -1]]
        label_batch = batch[-1].to(self.trainer_config.device)

        return input_batch, label_batch

    def _train_loop(
            self, feature_label_dataloader, use_sample_weights: bool = False,
            draw_plot: bool = False
    ):
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
                    loss = (loss_no_reduction * sample_weights).mean()
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

                # print epoch loss and accuracy after each 10% of training is done
                try:
                    if steps % (int(round(len(feature_label_dataloader) / 10))) == 0:
                        logger.info(f"Train loss: {epoch_loss / steps:.3f}, Train accuracy: {epoch_acc / steps:.3f}")
                except ZeroDivisionError:
                    continue

            avg_loss = epoch_loss / len(feature_label_dataloader)
            avg_acc = epoch_acc / len(feature_label_dataloader)
            train_losses.append(avg_loss)
            train_acc.append(avg_acc)

            logger.info("Epoch train loss: {}".format(avg_loss))
            logger.info("Epoch train accuracy: {}".format(avg_acc))

            if self.dev_model_input_x is not None:
                dev_clf_report, dev_loss = self.test(
                    self.dev_model_input_x, self.dev_gold_labels_y, loss_calculation=True)
                dev_losses.append(dev_loss)
                dev_acc.append(dev_clf_report["accuracy"])
                logger.info("Epoch development accuracy: {}".format(dev_clf_report["accuracy"]))

            # saving model
            if self.trainer_config.saved_models_dir is not None:
                model_path = os.path.join(
                    self.trainer_config.saved_models_dir,
                    f"model_state_dict_epoch_{current_epoch}.pt"
                )
                torch.save(self.model.cpu().state_dict(), model_path)
                self.model.to(self.trainer_config.device)

        log_section("Training done", logger)

        if draw_plot:
            if self.dev_model_input_x:
                draw_loss_accuracy_plot(
                    {"train loss": train_losses, "train acc": train_acc, "dev loss": dev_losses, "dev acc": dev_acc}
                )
            else:
                draw_loss_accuracy_plot({"train loss": train_losses, "train acc": train_acc})

        self.model.eval()


    def _prediction_loop(
            self, feature_label_dataloader: DataLoader, loss_calculation: str = False
    ) -> [np.ndarray, np.ndarray]:

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

        predictions = np.squeeze(np.hstack(predictions_list))
        gold_labels = np.squeeze(np.hstack(label_list))

        return predictions, gold_labels, dev_loss

    def _calculate_dev_loss(self, predictions: Tensor, labels: Tensor) -> Tensor:
        """ Calculates the loss on the dev set using given criterion"""
        predictions_one_hot = F.one_hot(predictions.argmax(1), num_classes=self.trainer_config.output_classes).float()
        labels_one_hot = F.one_hot(labels, self.trainer_config.output_classes)
        loss = self.trainer_config.criterion(predictions_one_hot, labels_one_hot)
        return loss.detach()

    def test(
            self, features_dataset: TensorDataset, labels: TensorDataset, loss_calculation: bool = False
    ) -> Tuple[Dict, Union[float, None]]:

        gold_labels = labels.tensors[0].cpu().numpy()

        if isinstance(self.model, skorch.NeuralNetClassifier):
            # when the pytorch model is wrapped as a sklearn model (e.g. cleanlab)
            predictions = self.model.predict(dataset_to_numpy_input(features_dataset))
        else:
            feature_label_dataset = input_labels_to_tensordataset(features_dataset, gold_labels)
            feature_label_dataloader = self._make_dataloader(feature_label_dataset, shuffle=False)
            predictions, gold_labels, dev_loss = self._prediction_loop(feature_label_dataloader, loss_calculation)

        if self.trainer_config.evaluate_with_other_class:
            clf_report = classification_report_other_class(
                y_true=gold_labels, y_pred=predictions, ids2labels=self.trainer_config.ids2labels,
                other_class_id=self.trainer_config.other_class_id
            )
        else:
            clf_report = classification_report(y_true=gold_labels, y_pred=predictions, output_dict=True)

        if loss_calculation:
            return clf_report, dev_loss / len(feature_label_dataloader)
        else:
            return clf_report, None
