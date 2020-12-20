import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader

from knodle.trainer.crossweight_denoising import utils
from knodle.trainer.crossweight_denoising.crossweight_weights_calculator import CrossWeightWeightsCalculator
from knodle.trainer.ds_model_trainer.ds_model_trainer import DsModelTrainer

from knodle.trainer.config.trainer_config import TrainerConfig
from knodle.trainer.config.crossweight_denoising_config import CrossWeightDenoisingConfig

# Set the seed for reproducibility
np.random.seed(12345)
torch.manual_seed(12345)
torch.cuda.manual_seed(12345)

disable_cuda = True
device = None
if not disable_cuda and torch.cuda.is_available():
    print("Using GPU")
    device = torch.device('cuda')
else:
    print("Using CPU")
    device = torch.device('cpu')

GRADIENT_CLIPPING = 5
PRINT_EVERY = 10


class CrossWeightTrainer(DsModelTrainer):

    def __init__(self,
                 model: Module,
                 rule_assignments_t: np.ndarray,
                 inputs_x: TensorDataset,
                 rule_matches_z: np.ndarray,
                 dev_inputs: TensorDataset,
                 dev_labels: np.ndarray,
                 trainer_config: TrainerConfig = None,
                 denoising_config: CrossWeightDenoisingConfig = None):
        """
        :param inputs_x: encoded samples (samples x features)
        :param rule_matches_z: binary matrix that contains info about rules matched in samples (samples x rules)
        :param rule_assignments_t: binary matrix that contains info about which rule correspond to which label
        :param model: a pre-defined classifier model that is to be trained
        :param denoising_config: config used for CrossWeight denoising
        :param trainer_config: config used for main training
        """
        super().__init__(
            model, rule_assignments_t, inputs_x, rule_matches_z, trainer_config
        )

        self.inputs_x = inputs_x
        self.rule_matches_z = rule_matches_z
        self.rule_assignments_t = rule_assignments_t

        self.dev_inputs = dev_inputs
        self.dev_labels = dev_labels

        if denoising_config is None:
            self.denoising_config = CrossWeightDenoisingConfig(self.model)
            self.logger.info(
                "Default CrossWeight Config is used: {}".format(self.denoising_config.__dict__)
            )
        else:
            self.denoising_config = denoising_config
            self.logger.info(
                "Initalized trainer with custom model config: {}".format(
                    self.denoising_config.__dict__
                )
            )

    def train(self):
        """
        This function weights the samples with CrossWeight method and train the model
        """
        # calculate sample weights
        sample_weights = CrossWeightWeightsCalculator(
            self.model, self.rule_assignments_t, self.inputs_x, self.rule_matches_z, self.denoising_config
        ).calculate_weights()

        self.logger.info("Classifier training is started")

        labels = utils.get_labels(self.rule_matches_z, self.rule_assignments_t)
        train_loader = self.convert2tensor(self.inputs_x, labels, sample_weights)
        dev_loader = self.convert2tensor(self.dev_inputs, self.dev_labels)

        self.model.train()
        steps_counter = 0

        for curr_epoch in range(self.trainer_config.epochs):
            for tokens, labels, weights in train_loader:
                tokens, labels, weights = tokens.to(device=device), labels.to(device=device), weights.to(device=device)
                steps_counter += 1
                tokens, labels = tokens.to(device=device), labels.to(device=device)

                self.trainer_config.criterion.reduction = 'none'
                self.trainer_config.optimizer.zero_grad()

                output = self.model(tokens)
                loss = self.get_train_loss(output, labels, weights)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIPPING)
                self.trainer_config.optimizer.step()

                if steps_counter % PRINT_EVERY == 0:
                    dev_loss = self.evaluate(dev_loader)
                    self.logger.info("Epoch: {}/{}...   Step: {}...   Loss: {:.6f}   Val Loss: {:.6f}".format(
                        curr_epoch + 1, self.trainer_config.epochs, steps_counter, loss.item(), dev_loss))
                    self.model.train()

    def evaluate(self, dev_loader):
        """ Model evaluation on dev set"""
        self.model.eval()
        with torch.no_grad():
            val_losses = []
            for tokens, labels in dev_loader:
                tokens, labels = tokens.to(device=device), labels.to(device=device)
                self.trainer_config.criterion.reduction = 'mean'
                output = self.model(tokens)
                val_loss = self.trainer_config.criterion(output, labels)
                val_losses.append(val_loss.item())
        return np.mean(val_losses)

    def convert2tensor(
            self, samples: TensorDataset, labels: np.ndarray, sample_weights: np.ndarray = None, shuffle: bool = True
    ) -> DataLoader:
        """ Converts encoded samples and labels to dataloader. Optionally: sample weights (in train dataloader) """

        tensor_target = torch.LongTensor(labels).to(device=device)
        tensor_samples = samples.tensors[0].long().to(device=device)

        if sample_weights is not None:
            sample_weights = torch.FloatTensor(sample_weights).to(device=device)
            dataset = torch.utils.data.TensorDataset(tensor_samples, tensor_target, sample_weights)
        else:
            dataset = torch.utils.data.TensorDataset(tensor_samples, tensor_target)

        return torch.utils.data.DataLoader(dataset, batch_size=self.trainer_config.batch_size, shuffle=shuffle)

    def get_train_loss(self, output, labels, weights):
        return (self.trainer_config.criterion(output, labels) * weights).sum() / \
               self.trainer_config.class_weights[labels].sum()
