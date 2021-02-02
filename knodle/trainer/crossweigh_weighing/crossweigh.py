import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.autograd import function
from torch.functional import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader
from joblib import load
from tqdm import tqdm
import torch.nn.functional as F
import os
from typing import Dict, Tuple, Union

from knodle.trainer.crossweigh_weighing.crossweigh_denoising_config import CrossWeighDenoisingConfig
from knodle.trainer.crossweigh_weighing.crossweigh_trainer_config import CrossWeighTrainerConfig
from knodle.trainer.crossweigh_weighing.utils import (
    set_seed, make_plot, get_labels, calculate_dev_tacred_metrics
)
from knodle.trainer.crossweigh_weighing.crossweigh_weights_calculator import CrossWeighWeightsCalculator
from knodle.trainer.trainer import Trainer
from knodle.trainer.utils.utils import accuracy_of_probs

torch.set_printoptions(edgeitems=100)
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True


class CrossWeigh(Trainer):

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
            
        set_seed(self.trainer_config.seed)

    def train(self):
        """ This function sample_weights the samples with CrossWeigh method and train the model """

        sample_weights = self._get_sample_weights() if self.use_weights \
            else torch.FloatTensor([1] * len(self.model_input_x))

        if not self.run_classifier:
            logger.info("No classifier should be trained")
            return

        logger.info("Classifier training is started")

        train_labels = get_labels(
            self.rule_matches_z, self.rule_assignments_t, self.trainer_config.no_match_class_label)
        train_loader = self._get_feature_label_dataloader(self.model_input_x, train_labels, sample_weights)
        train_losses, train_acc = [], []

        if self.dev_features is not None:
            dev_loader = self._get_feature_label_dataloader(self.dev_features, self.dev_labels)
            dev_losses, dev_acc = [], []

        self.model.train()
        for curr_epoch in tqdm(range(self.trainer_config.epochs)):
            logger.info(f"Epoch {curr_epoch}")
            running_loss, epoch_acc = 0.0, 0.0
            self.trainer_config.criterion.weight = self.trainer_config.class_weights
            self.trainer_config.criterion.reduction = 'none'
            batch_losses = []
            for features, labels, weights in train_loader:
                self.model.zero_grad()
                predictions = self.model(features)
                loss = self._get_loss_with_sample_weights(self.trainer_config.criterion, predictions, labels, weights)
                loss.backward()
                if self.trainer_config.use_grad_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.grad_clipping)
                self.trainer_config.optimizer.step()
                acc = accuracy_of_probs(predictions, labels)

                running_loss += loss.detach()
                batch_losses.append(running_loss)
                epoch_acc += acc.item()

            avg_loss = running_loss / len(train_loader)
            avg_acc = epoch_acc / len(train_loader)
            train_losses.append(avg_loss)
            train_acc.append(avg_acc)
            logger.info(f"Train loss: {avg_loss:.7f}, train accuracy: {avg_acc * 100:.2f}%")

            if self.dev_features is not None:
                dev_loss, dev_metrics = self._evaluate(dev_loader)
                dev_losses.append(dev_loss)
                dev_acc.append(dev_metrics["precision"])
                logger.info(f"Dev loss: {dev_loss:.3f}, Dev metrics: {dev_metrics}")

        if self.dev_features is not None:
            make_plot({"train loss": train_losses, "dev loss": dev_losses, "tran acc": train_acc, "dev acc": dev_acc})
        else:
            make_plot({"train loss": train_losses, "tran acc": train_acc})

    def _get_sample_weights(self):
        """ This function checks whether there are accesible already pretrained sample weights. If yes, return
        them. If not, calculates sample weights calling method of CrossWeighWeightsCalculator class"""

        try:
            sample_weights = load(os.path.join(self.path_to_weights, "sample_weights.lib"))
            logger.info("Already pretrained samples sample_weights will be used.")
        except OSError:
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

    def _get_feature_label_dataloader(
            self, samples: TensorDataset, labels: Union[Tensor, np.ndarray], sample_weights: np.ndarray = None, shuffle: bool = True
    ) -> DataLoader:
        """ Converts encoded samples and labels to dataloader. Optionally: add sample_weights as well """
        tensor_target = torch.LongTensor(labels).to(self.trainer_config.device)
        tensor_samples = samples.tensors[0].to(self.trainer_config.device)

        if sample_weights is not None:
            sample_weights = torch.FloatTensor(sample_weights).to(self.trainer_config.device)
            dataset = torch.utils.data.TensorDataset(tensor_samples, tensor_target, sample_weights)
        else:
            dataset = torch.utils.data.TensorDataset(tensor_samples, tensor_target)

        return self._make_dataloader(dataset, shuffle=shuffle)

    def _get_loss_with_sample_weights(
            self, criterion: function, output: Tensor, labels: Tensor, weights: Tensor) -> Tensor:
        """ Calculates loss for each training sample and multiplies it with corresponding sample weight"""
        return (criterion(output, labels) * weights).sum() / self.trainer_config.class_weights[labels].sum()

    def _evaluate(self, dev_dataloader: DataLoader) -> Union[Tuple[float, None], Tuple[float, Dict]]:
        """ Model evaluation on dev set: the trained model is applied on the dev set and the average loss is returned"""
        self.model.eval()
        all_predictions, all_labels = torch.Tensor(), torch.Tensor()

        with torch.no_grad():
            dev_loss, dev_acc = 0.0, 0.0
            dev_criterion = nn.CrossEntropyLoss(weight=self.trainer_config.class_weights)
            for features, labels in dev_dataloader:
                predictions = self.model(features)
                dev_loss += self.calculate_dev_loss(predictions, labels.long(), dev_criterion)

                _, predicted = torch.max(predictions, 1)
                all_predictions = torch.cat([all_predictions, predicted])
                all_labels = torch.cat([all_labels, labels.long()])

            predictions, gold_labels = (all_predictions.detach().numpy(), all_labels.detach().numpy())
            dev_metrics = self.calculate_dev_metrics(predictions, gold_labels)
        return dev_loss / len(dev_dataloader), dev_metrics

    def calculate_dev_loss(self, predictions: Tensor, labels: Tensor, criterion: function) -> Tensor:
        """ Calculates the loss on the dev set using given criterion"""
        predictions_one_hot = F.one_hot(predictions.argmax(1), num_classes=self.trainer_config.output_classes).float()
        loss = criterion(predictions_one_hot, labels.flatten(0))
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
                return classification_report(y_true=gold_labels, y_pred=predictions, output_dict=True)["macro avg"]

            return calculate_dev_tacred_metrics(predictions, gold_labels, self.dev_labels_ids)

        elif self.evaluation_method == "sklearn_classification_report":
            return classification_report(y_true=gold_labels, y_pred=predictions, output_dict=True)["macro avg"]

        else:
            logging.warning("No evaluation method is given. The evaluation on dev data is skipped")
            return None
