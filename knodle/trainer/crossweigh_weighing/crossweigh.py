import logging
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import function
from torch.functional import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader
from joblib import load
from tqdm import tqdm

from knodle.trainer.config.crossweigh_denoising_config import CrossWeighDenoisingConfig
from knodle.trainer.config.crossweigh_trainer_config import CrossWeighTrainerConfig
from knodle.trainer.crossweigh_weighing.utils import set_device, set_seed
from knodle.trainer.crossweigh_weighing.crossweigh_weights_calculator import CrossWeighWeightsCalculator
from knodle.trainer.ds_model_trainer.ds_model_trainer import DsModelTrainer
from knodle.trainer.utils.denoise import get_majority_vote_probs, get_majority_vote_probs_with_no_rel
from knodle.trainer.utils.utils import accuracy_of_probs

PRINT_EVERY = 100
NO_MATCH_CLASS = -1
torch.set_printoptions(edgeitems=100)
logger = logging.getLogger(__name__)


class CrossWeigh(DsModelTrainer):

    def __init__(self,
                 model: Module,
                 rule_assignments_t: np.ndarray,
                 inputs_x: TensorDataset,
                 rule_matches_z: np.ndarray,
                 dev_features_labels: TensorDataset,
                 weights: np.ndarray = None,
                 denoising_config: CrossWeighDenoisingConfig = None,
                 trainer_config: CrossWeighTrainerConfig = None):
        """
        :param model: a pre-defined classifier model that is to be trained
        :param rule_assignments_t: binary matrix that contains info about which rule correspond to which label
        :param inputs_x: encoded samples (samples x features)
        :param rule_matches_z: binary matrix that contains info about rules matched in samples (samples x rules)
        :param dev_features_labels: development samples and corresponding labels used for model evaluation
        :param trainer_config: config used for main training
        :param denoising_config: config used for CrossWeigh denoising
        """
        super().__init__(
            model, rule_assignments_t, inputs_x, rule_matches_z, trainer_config
        )

        self.inputs_x = inputs_x
        self.rule_matches_z = rule_matches_z
        self.rule_assignments_t = rule_assignments_t
        self.weights = weights
        self.denoising_config = denoising_config
        self.dev_features_labels = dev_features_labels

        if trainer_config is None:
            self.trainer_config = CrossWeighTrainerConfig(self.model)
            logger.info("Default CrossWeigh Config is used: {}".format(self.trainer_config.__dict__))
        else:
            self.trainer_config = trainer_config
            logger.info("Initalized trainer with custom model config: {}".format(self.trainer_config.__dict__))

        self.device = set_device(self.trainer_config.enable_cuda)

    def train(self):
        """ This function sample_weights the samples with CrossWeigh method and train the model """
        set_seed(self.trainer_config.seed)

        if self.weights is not None:
            logger.info("Already pretrained samples sample_weights will be used.")
            sample_weights = load(self.weights)
        else:
            logger.info("No pretrained sample sample_weights are found, they will be calculated now")
            sample_weights = CrossWeighWeightsCalculator(
                self.model, self.rule_assignments_t, self.inputs_x, self.rule_matches_z, self.denoising_config
            ).calculate_weights()

        logger.info("Classifier training is started")

        if self.trainer_config.negative_samples:
            labels = get_majority_vote_probs_with_no_rel(self.rule_matches_z, self.rule_assignments_t, NO_MATCH_CLASS)
        else:
            labels = get_majority_vote_probs(self.rule_matches_z, self.rule_assignments_t)

        train_loader = self._get_feature_label_dataloader(self.inputs_x, labels, sample_weights)

        self.model.train()

        for curr_epoch in tqdm(range(self.trainer_config.epochs)):
            epoch_loss, epoch_acc = 0.0, 0.0
            self.trainer_config.criterion.weight = self.trainer_config.class_weights
            self.trainer_config.criterion.reduction = 'none'
            batch_counter = 0
            for feature_batch, label_batch, weights_batch in train_loader:
                batch_counter += 1
                self.model.zero_grad()
                predictions = self.model(feature_batch)
                # loss = self.trainer_config.criterion(predictions, label_batch)
                loss = self._get_train_loss(self.trainer_config.criterion, predictions, label_batch, weights_batch)
                loss.backward()

                if self.trainer_config.use_grad_clipping:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.grad_clipping)

                self.trainer_config.optimizer.step()
                acc = accuracy_of_probs(predictions, label_batch)

                epoch_loss += loss.detach()
                epoch_acc += acc.item()

            avg_loss = epoch_loss / len(train_loader)
            avg_acc = epoch_acc / len(train_loader)

            logger.info("Epoch loss: {}".format(avg_loss))
            logger.info("Epoch Accuracy: {}".format(avg_acc))

            # self._print_intermediate_result(steps_counter, curr_epoch, loss, dev_loader)

    def _evaluate(self, dev_loader):
        """ Model evaluation on dev set: the trained model is applied on the dev set and the average loss value
        is returned """
        self.model.eval()
        dev_criterion = nn.CrossEntropyLoss(weight=self.trainer_config.class_weights)
        with torch.no_grad():
            dev_losses = []
            for tokens, labels in dev_loader:
                tokens, labels = tokens.to(device=self.device), labels.to(device=self.device)
                output = self.model(tokens)
                dev_loss = dev_criterion(output, labels)
                dev_losses.append(dev_loss.item())
        return np.mean(dev_losses)

    def _get_feature_label_dataloader(
            self, samples: TensorDataset, labels: np.ndarray, sample_weights: np.ndarray = None, shuffle: bool = True
    ) -> DataLoader:
        """ Converts encoded samples and labels to dataloader. Optionally: add sample_weights as well """

        tensor_target = torch.LongTensor(labels).to(device=self.device)
        tensor_samples = samples.tensors[0].long().to(device=self.device)
        if sample_weights is not None:
            sample_weights = torch.FloatTensor(sample_weights).to(device=self.device)
            dataset = torch.utils.data.TensorDataset(tensor_samples, tensor_target, sample_weights)
        else:
            dataset = torch.utils.data.TensorDataset(tensor_samples, tensor_target)
        dataloader = self._make_dataloader(dataset, shuffle=shuffle)
        return dataloader

    def _get_train_loss(self, criterion: function, output: Tensor, labels: Tensor, weights: Tensor) -> Tensor:
        return (criterion(output, labels) * weights).sum() / self.trainer_config.class_weights[labels].sum()

    def _print_intermediate_result(
            self, curr_step: int, curr_epoch: int, curr_loss: Tensor, dev_loader: DataLoader
    ) -> None:
        if curr_step % PRINT_EVERY == 0:
            dev_loss = self._evaluate(dev_loader)
            self.test(TensorDataset(self.dev_features_labels.tensors[0]),
                      TensorDataset(self.dev_features_labels.tensors[1]))
            logger.info("Epoch: {}/{}...   Step: {}...   Train Loss: {:.6f}     Dev Loss: {:.6f}".format(
                curr_epoch + 1, self.trainer_config.epochs, curr_step, curr_loss.item(), dev_loss))

