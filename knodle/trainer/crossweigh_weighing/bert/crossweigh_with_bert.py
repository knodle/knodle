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
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from knodle.trainer.baseline.no_denoising import NoDenoisingTrainer
from knodle.trainer.crossweigh_weighing.crossweigh_denoising_config import CrossWeighDenoisingConfig
from knodle.trainer.crossweigh_weighing.crossweigh_trainer_config import CrossWeighTrainerConfig
from knodle.trainer.crossweigh_weighing.bert.crossweigh_weights_calculator_bert import CrossWeighWeightsCalculator
from knodle.trainer.crossweigh_weighing.utils import (
    set_seed, draw_loss_accuracy_plot, get_labels, calculate_dev_tacred_metrics, build_bert_feature_labels_dataloader,
    build_bert_feature_weights_labels_dataloader
)

from knodle.transformation.filter import filter_empty_probabilities_x_y_z
from knodle.trainer.utils.utils import accuracy_of_probs

torch.set_printoptions(edgeitems=100)
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True

PRINT_EVERY = 300
SAVE_DIR = "/Users/asedova/PycharmProjects/knodle/knodle/trainer/crossweigh_weighing/bert/run_10_02"


class CrossWeigh(NoDenoisingTrainer):

    def __init__(
            self,
            model: Module,
            rule_assignments_t: np.ndarray,
            inputs_x: TensorDataset,
            rule_matches_z: np.ndarray,
            dev_features: TensorDataset = None,
            dev_labels: Tensor = None,
            evaluation_method: str = "sklearn_classification_report",
            dev_labels_ids: Dict = None,
            path_to_weights: str = "data",
            denoising_config: CrossWeighDenoisingConfig = None,
            trainer_config: CrossWeighTrainerConfig = None,
            run_classifier: bool = True,
            use_weights: bool = True
    ):
        """
        :param model: a pre-defined classifier model that is to be trained
        :param rule_assignments_t: binary matrix that contains info about which rule correspond to which label
        :param inputs_x: encoded samples (samples x features)
        :param rule_matches_z: binary matrix that contains info about rules matched in samples (samples x rules)
        :param trainer_config: config used for main training
        :param denoising_config: config used for CrossWeigh denoising
        """
        super().__init__(
            model, rule_assignments_t, inputs_x, rule_matches_z, trainer_config
        )

        self.inputs_x = inputs_x
        self.rule_matches_z = rule_matches_z
        self.rule_assignments_t = rule_assignments_t
        self.denoising_config = denoising_config
        self.dev_features = dev_features
        self.dev_labels = dev_labels
        self.evaluation_method = evaluation_method
        self.dev_labels_ids = dev_labels_ids
        self.path_to_weights = path_to_weights
        self.run_classifier = run_classifier
        self.use_weights = use_weights

        if trainer_config is None:
            self.trainer_config = CrossWeighTrainerConfig(self.model)
            logger.info("Default CrossWeigh Config is used: {}".format(self.trainer_config.__dict__))
        else:
            self.trainer_config = trainer_config
            logger.info("Initalized trainer with custom model config: {}".format(self.trainer_config.__dict__))

    def train(self):
        """ This function sample_weights the samples with CrossWeigh method and train the model """

        set_seed(self.trainer_config.seed)

        train_labels = get_labels(
            self.rule_matches_z, self.rule_assignments_t, self.trainer_config.no_match_class_label)

        if self.trainer_config.filter_empty_probs:
            self.inputs_x, self.rule_matches_z, train_labels = filter_empty_probabilities_x_y_z(
                self.inputs_x, train_labels, self.rule_matches_z
            )

        sample_weights = self._get_sample_weights() if self.use_weights \
            else torch.FloatTensor([1] * len(self.inputs_x))

        if not self.run_classifier:
            logger.info("No classifier should be trained")
            return
        logger.info("Classifier training is started")

        train_loader = build_bert_feature_weights_labels_dataloader(
            self.inputs_x, sample_weights, train_labels, self.trainer_config.batch_size
        )

        if self.dev_features:
            dev_loader = build_bert_feature_labels_dataloader(
                self.dev_features, self.dev_labels, self.trainer_config.batch_size
            )
            dev_losses, dev_acc = [], []

        train_losses, train_acc = [], []
        self.model.train()
        for curr_epoch in range(self.trainer_config.epochs):
            logger.info(f"Epoch {curr_epoch}")

            os.makedirs(SAVE_DIR, exist_ok=True)
            path_to_saved_model = os.path.join(SAVE_DIR, 'model_epoch_{}.pth'.format(curr_epoch))
            steps = 0

            running_loss, epoch_acc = 0.0, 0.0
            for input_ids_batch, attention_mask_batch, weights, labels in tqdm(train_loader):
                steps += 1
                inputs = {
                    "input_ids": input_ids_batch.to(self.trainer_config.device),
                    "attention_mask": attention_mask_batch.to(self.trainer_config.device)
                }
                weights, labels = weights.to(self.trainer_config.device), labels.to(self.trainer_config.device)

                self.model.zero_grad()
                predictions = self.model(**inputs)
                loss = self._get_loss_with_sample_weights(predictions[0], weights, labels)
                loss.backward()
                if self.trainer_config.use_grad_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.grad_clipping)
                self.trainer_config.optimizer.step()
                acc = accuracy_of_probs(predictions[0], labels)

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

            if self.dev_features:
                dev_loss, dev_metrics = self._evaluate(dev_loader)
                dev_losses.append(dev_loss)
                dev_acc.append(dev_metrics["precision"])
                logger.info(f"Dev loss: {dev_loss:.3f}, Dev metrics: {dev_metrics}")

            torch.save(self.model.cpu().state_dict(), path_to_saved_model)  # saving model
            self.model.to(self.trainer_config.device)

        if self.dev_features:
            draw_loss_accuracy_plot(
                {"train loss": train_losses, "dev loss": dev_losses, "tran acc": train_acc, "dev acc": dev_acc})
        else:
            draw_loss_accuracy_plot({"train loss": train_losses, "tran acc": train_acc})

    def _get_sample_weights(self):
        """ This function checks whether there are accesible already pretrained sample weights. If yes, return
        them. If not, calculates sample weights calling method of CrossWeighWeightsCalculator class"""

        if os.path.isfile(os.path.join(self.path_to_weights, "sample_weights.lib")):
            logger.info("Already pretrained samples sample_weights will be used.")
            sample_weights = load(os.path.join(self.path_to_weights, "sample_weights.lib"))
        else:
            logger.info("No pretrained sample weights are found, they will be calculated now")
            sample_weights = CrossWeighWeightsCalculator(
                self.model,
                self.rule_assignments_t,
                self.inputs_x,
                self.rule_matches_z,
                self.path_to_weights,
                self.denoising_config
            ).calculate_weights()
            logger.info(f"Sample weights are calculated and saved to {self.path_to_weights} file")
        return sample_weights

    # todo: move to utils
    def _get_loss_with_sample_weights(self, output: Tensor, weights: Tensor, labels: Tensor) -> Tensor:
        """ Calculates loss for each training sample and multiplies it with corresponding sample weight"""
        loss_no_reduction = self.trainer_config.criterion(output,
                                                          labels,
                                                          weight=self.trainer_config.class_weights,
                                                          reduction="none")
        return (loss_no_reduction * weights).mean()
        # normalisation of the sample weights so that the range of the loss will approx. have the same range and
        # won’t depend on the current sample distribution in the batch.
        # return (loss_no_reduction * weights / weights.sum()).sum()

    def _evaluate(self, dev_dataloader: DataLoader) -> Union[Tuple[float, None], Tuple[float, Dict]]:
        """ Model evaluation on dev set: the trained model is applied on the dev set and the average loss is returned"""
        self.model.eval()
        all_predictions, all_labels = torch.Tensor().to(self.trainer_config.device), torch.Tensor().to(self.trainer_config.device)

        with torch.no_grad():
            dev_loss, dev_acc = 0.0, 0.0
            for input_ids_batch, attention_mask_batch, labels in dev_dataloader:
                inputs = {
                    "input_ids": input_ids_batch.to(self.trainer_config.device),
                    "attention_mask": attention_mask_batch.to(self.trainer_config.device)
                }
                labels = labels.to(self.trainer_config.device)

                self.model.zero_grad()
                predictions = self.model(**inputs)
                dev_loss += self.calculate_dev_loss(predictions[0], labels.long())

                _, predicted = torch.max(predictions[0], 1)
                all_predictions = torch.cat([all_predictions, predicted])
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

