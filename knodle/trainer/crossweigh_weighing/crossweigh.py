import logging
import os
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from joblib import load
from sklearn.metrics import classification_report
from torch.functional import Tensor
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from knodle.trainer.baseline.no_denoising import NoDenoisingTrainer
from knodle.trainer.crossweigh_weighing.config import CrossWeighDenoisingConfig
from knodle.trainer.crossweigh_weighing.crossweigh_weights_calculator import CrossWeighWeightsCalculator
from knodle.trainer.crossweigh_weighing.utils import (
    draw_loss_accuracy_plot, get_labels, calculate_dev_tacred_metrics,
    build_feature_weights_labels_dataloader, build_feature_labels_dataloader
)

from knodle.transformation.filter import filter_empty_probabilities_x_y_z
from knodle.trainer.utils.utils import accuracy_of_probs

torch.set_printoptions(edgeitems=100)
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True

PRINT_EVERY = 300
SAVE_DIR = "/Users/asedova/PycharmProjects/knodle/knodle/trainer/crossweigh_weighing/bert/run_10_02"


class CrossWeighTrainer(NoDenoisingTrainer):

    def __init__(
            self,
            cw_model: Module = None,
            cw_model_input_x: TensorDataset = None,
            evaluation_method: str = "sklearn_classification_report",
            dev_labels_ids: Dict = None,
            path_to_weights: str = "data",
            run_classifier: bool = True,
            use_weights: bool = True,
            **kwargs
    ):
        """
        :param model: a pre-defined classifier model that is to be trained
        :param rule_assignments_t: binary matrix that contains info about which rule correspond to which label
        :param inputs_x: encoded samples (samples x features)
        :param rule_matches_z: binary matrix that contains info about rules matched in samples (samples x rules)
        """
        self.cw_model = cw_model if cw_model else kwargs.get("model")
        self.cw_model_input_x = cw_model_input_x if cw_model_input_x else kwargs.get("model_input_x")

        if kwargs.get("trainer_config") is None:
            kwargs["trainer_config"] = CrossWeighDenoisingConfig(
                optimizer=SGD(kwargs.get("model").parameters(), lr=0.001),
                cw_optimizer=SGD(self.cw_model.parameters(), lr=0.001)
            )
        super().__init__(**kwargs)

        self.evaluation_method = evaluation_method
        self.dev_labels_ids = dev_labels_ids
        self.path_to_weights = path_to_weights
        self.run_classifier = run_classifier
        self.use_weights = use_weights

    def train(self):
        """ This function sample_weights the samples with CrossWeigh method and train the model """

        train_labels = self.calculate_labels()

        sample_weights = self._get_sample_weights() if self.use_weights \
            else torch.FloatTensor([1] * len(self.model_input_x))

        if not self.run_classifier:
            logger.info("No classifier should be trained")
            return
        logger.info("Classifier training is started")

        train_loader = build_feature_weights_labels_dataloader(
            self.model_input_x, sample_weights, train_labels, self.trainer_config.batch_size
        )

        if self.dev_model_input_x:
            dev_loader = build_feature_labels_dataloader(
                self.dev_model_input_x, self.dev_gold_labels_y, self.trainer_config.batch_size
            )
            dev_losses, dev_acc = [], []

        train_losses, train_acc = [], []

        self.model.to(self.trainer_config.device)
        self.model.train()

        for curr_epoch in range(self.trainer_config.epochs):
            logger.info(f"Epoch {curr_epoch}")

            path_to_saved_model = os.path.join(self.path_to_weights, "trained_models")
            os.makedirs(path_to_saved_model, exist_ok=True)
            path_to_saved_model = os.path.join(path_to_saved_model, f'model_epoch_{curr_epoch}.pth')

            steps = 0

            running_loss, epoch_acc = 0.0, 0.0
            for batch in tqdm(train_loader):
                features, labels = self._load_batch(batch)
                steps += 1

                self.trainer_config.optimizer.zero_grad()
                self.model.zero_grad()

                predictions = self.model(*features)
                predictions = predictions[0] if not isinstance(predictions, torch.Tensor) else predictions
                loss = self._get_loss_with_sample_weights(predictions, features[-1], labels)
                loss.backward()
                if isinstance(self.trainer_config.grad_clipping, int):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.grad_clipping)
                self.trainer_config.optimizer.step()
                acc = accuracy_of_probs(predictions, labels)

                running_loss += loss.detach()
                epoch_acc += acc.item()

                if steps % PRINT_EVERY == 0 and self.dev_features:
                    dev_loss, dev_metrics = self._evaluate(dev_loader)
                    logger.info(f"Train loss: {running_loss / steps:.3f}, Train accuracy: {epoch_acc / steps:.3f}, "
                                f"Dev loss: {dev_loss:.3f}, Dev metrics: {dev_metrics}")

            avg_loss = running_loss / len(train_loader)
            avg_acc = epoch_acc / len(train_loader)
            train_losses.append(avg_loss)
            train_acc.append(avg_acc)
            logger.info(f"Train loss: {avg_loss:.7f}, train accuracy: {avg_acc * 100:.2f}%")

            if self.dev_model_input_x is not None:
                dev_loss, dev_metrics = self._evaluate(dev_loader)
                dev_losses.append(dev_loss)
                dev_acc.append(dev_metrics["accuracy"])
                logger.info(f"Dev loss: {dev_loss:.3f}, Dev metrics: {dev_metrics}")

        torch.save(self.model.cpu().state_dict(), path_to_saved_model)  # saving model
        self.model.to(self.trainer_config.device)

        if self.dev_model_input_x is not None:
            draw_loss_accuracy_plot(
                {"train loss": train_losses, "dev loss": dev_losses, "tran acc": train_acc, "dev acc": dev_acc})
        else:
            draw_loss_accuracy_plot({"train loss": train_losses, "tran acc": train_acc})

    def calculate_labels(self) -> np.ndarray:

        if not self.trainer_config.filter_non_labelled and self.trainer_config.other_class_id is None:
            self.trainer_config.other_class_id = self.mapping_rules_labels_t.shape[1]

        train_labels = get_labels(
            self.rule_matches_z, self.mapping_rules_labels_t, self.trainer_config.other_class_id
        )

        if self.trainer_config.filter_non_labelled:
            self.model_input_x, self.rule_matches_z, train_labels = filter_empty_probabilities_x_y_z(
                self.model_input_x, train_labels, self.rule_matches_z
            )

        if train_labels.shape[1] != self.trainer_config.output_classes:
            raise ValueError(
                f"The number of output classes {self.trainer_config.output_classes} do not correspond to labels "
                f"probabilities dimension {train_labels.shape[1]}"
            )

        return train_labels

    def _get_sample_weights(self):
        """ This function checks whether there are accessible already pretrained sample weights. If yes, return
        them. If not, calculates sample weights calling method of CrossWeighWeightsCalculator class"""

        if os.path.isfile(os.path.join(self.path_to_weights, "sample_weights.lib")):
            logger.info("Already pretrained samples sample_weights will be used.")
            sample_weights = load(os.path.join(self.path_to_weights, "sample_weights.lib"))
        else:
            logger.info("No pretrained sample weights are found, they will be calculated now")
            sample_weights = CrossWeighWeightsCalculator(
                self.cw_model,
                self.mapping_rules_labels_t,
                self.cw_model_input_x,
                self.rule_matches_z,
                self.path_to_weights,
                self.trainer_config
            ).calculate_weights()
            logger.info(f"Sample weights are calculated and saved to {self.path_to_weights} file")
        return sample_weights

    # def _get_feature_label_dataloader(
    #         self, samples: TensorDataset, labels: Union[TensorDataset, Tensor, np.ndarray],
    #         sample_weights: np.ndarray = None, shuffle: bool = True
    # ) -> DataLoader:
    #     """ Converts encoded samples and labels to dataloader. Optionally: add sample_weights as well """
    #     if type(labels) is TensorDataset:
    #         labels = labels.tensors[0]
    #     tensor_target = torch.LongTensor(labels).to(self.trainer_config.device)
    #     tensor_samples = samples.tensors[0].to(self.trainer_config.device)
    #
    #     if sample_weights is not None:
    #         sample_weights = torch.FloatTensor(sample_weights).to(self.trainer_config.device)
    #         dataset = torch.utils.data.TensorDataset(tensor_samples, tensor_target, sample_weights)
    #     else:
    #         dataset = torch.utils.data.TensorDataset(tensor_samples, tensor_target)
    #
    #     return self._make_dataloader(dataset, shuffle=shuffle)

    def _get_loss_with_sample_weights(self, output: Tensor, weights: Tensor, labels: Tensor) -> Tensor:
        """ Calculates loss for each training sample and multiplies it with corresponding sample weight"""
        loss_no_reduction = self.trainer_config.criterion(
            output, labels, weight=self.trainer_config.class_weights, reduction="none"
        )
        return (loss_no_reduction * weights).mean()

    def _evaluate(self, dev_dataloader: DataLoader) -> Union[Tuple[float, None], Tuple[float, Dict]]:
        """ Model evaluation on dev set: the trained model is applied on the dev set and the average loss is returned"""
        self.model.eval()
        all_predictions, all_labels = torch.Tensor().to(self.trainer_config.device), \
                                      torch.Tensor().to(self.trainer_config.device)

        with torch.no_grad():
            dev_loss, dev_acc = 0.0, 0.0
            for batch in dev_dataloader:
                features, labels = self._load_batch(batch)
                self.trainer_config.optimizer.zero_grad()

                self.model.zero_grad()
                predictions = self.model(*features)
                predictions = predictions[0] if not isinstance(predictions, torch.Tensor) else predictions
                dev_loss += self.calculate_dev_loss(predictions, labels.long())

                _, predicted = torch.max(predictions, 1)
                all_predictions = torch.cat([all_predictions, predicted.float()])
                all_labels = torch.cat([all_labels, labels.long()])

            predictions, gold_labels = (all_predictions.cpu().detach().numpy(), all_labels.cpu().detach().numpy())
            dev_metrics = self.calculate_dev_metrics(predictions, gold_labels)
        return dev_loss / len(dev_dataloader), dev_metrics

    def calculate_dev_loss(self, predictions: Tensor, labels: Tensor) -> Tensor:
        """ Calculates the loss on the dev set using given criterion"""
        predictions_one_hot = F.one_hot(predictions.argmax(1), num_classes=self.trainer_config.output_classes).float()
        labels_one_hot = F.one_hot(labels, self.trainer_config.output_classes)
        loss = self.trainer_config.criterion(predictions_one_hot, labels_one_hot)
        return loss.detach()

    def calculate_dev_metrics(self, predictions: np.ndarray, gold_labels: np.ndarray) -> Union[Dict, None]:
        """
        Returns the dictionary of metrics calculated on the dev set with one of the evaluation functions
        or None, if the needed evaluation method was not found
        """

        if self.evaluation_method == "tacred":
            if self.dev_labels_ids is None:
                logging.warning(
                    "Labels to labels ids correspondence is needed to make TACRED specific evaluation. Since it is "
                    "absent now, the standard sklearn metrics will be calculated instead"
                )
                return classification_report(y_true=gold_labels, y_pred=predictions, output_dict=True)

            return calculate_dev_tacred_metrics(predictions, gold_labels, self.dev_labels_ids)

        elif self.evaluation_method == "sklearn_classification_report":
            return classification_report(y_true=gold_labels, y_pred=predictions, output_dict=True)

        else:
            logging.warning("No evaluation method is given. The evaluation on dev data is skipped")
            return None
