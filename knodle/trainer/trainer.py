import logging
from typing import Dict
from tqdm.auto import tqdm
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import classification_report

import torch
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader

from knodle.transformation.torch_input import input_labels_to_tensordataset
from knodle.trainer.config import TrainerConfig
from knodle.trainer.utils.utils import log_section, accuracy_of_probs

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
    def train(self, model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None):
        pass

    @abstractmethod
    def test(self, test_features: TensorDataset, test_labels: TensorDataset):
        pass


class BaseTrainer(Trainer):

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

    def _train_loop(self, feature_label_dataloader):
        log_section("Training starts", logger)

        self.model.to(self.trainer_config.device)
        self.model.train()
        for current_epoch in tqdm(range(self.trainer_config.epochs)):
            epoch_loss, epoch_acc = 0.0, 0.0
            logger.info("Epoch: {}".format(current_epoch))

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

            avg_loss = epoch_loss / len(feature_label_dataloader)
            avg_acc = epoch_acc / len(feature_label_dataloader)

            logger.info("Epoch train loss: {}".format(avg_loss))
            logger.info("Epoch train accuracy: {}".format(avg_acc))

            if self.dev_model_input_x is not None:
                clf_report = self.test(self.dev_model_input_x, self.dev_gold_labels_y)
                logger.info("Epoch development accuracy: {}".format(clf_report["accuracy"]))

        log_section("Training done", logger)

        self.model.eval()

    def _prediction_loop(self, feature_label_dataloader: DataLoader) -> [np.ndarray, np.ndarray]:
        self.model.to(self.trainer_config.device)
        self.model.eval()
        predictions_list, label_list = [], []

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

        # transform to correct format.
        predictions = np.squeeze(np.hstack(predictions_list))
        gold_labels = np.squeeze(np.hstack(label_list))

        return predictions, gold_labels

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

    def test(self, test_features: TensorDataset, test_labels: TensorDataset) -> Dict:
        """
        Runs evaluation and returns a classification report with different evaluation metrics.
        Args:
            test_features: Test feature set
            test_labels: Gold label set.

        Returns:

        """
        feature_label_dataset = input_labels_to_tensordataset(test_features, test_labels.tensors[0].cpu().numpy())
        feature_label_dataloader = self._make_dataloader(feature_label_dataset, shuffle=False)

        predictions, gold_labels = self._prediction_loop(feature_label_dataloader)

        clf_report = classification_report(y_true=gold_labels, y_pred=predictions, output_dict=True)

        return clf_report
