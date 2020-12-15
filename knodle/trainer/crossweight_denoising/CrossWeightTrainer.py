import logging
from datetime import date

from knodle.model.BidirectionalLSTM.BidirectionalLSTM import BidirectionalLSTM
from knodle.trainer.config.TrainerConfig import TrainerConfig
from knodle.trainer.crossweight_denoising import utils
from knodle.trainer.ds_model_trainer.ds_model_trainer import DsModelTrainer
import torch
import torch.nn as nn
from torch.nn import Module
import os
from knodle.trainer.config.CrossWeightDenoisingConfig import CrossWeightDenoisingConfig

import numpy as np
from torch.utils.data import TensorDataset, DataLoader

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

logger = logging.getLogger(__name__)
GRADIENT_CLIPPING = 5


class CrossWeightTrainer(DsModelTrainer):

    def __init__(self,
                 model: Module,
                 denoising_config: CrossWeightDenoisingConfig = None,
                 trainer_config: TrainerConfig = None):
        super().__init__(model, trainer_config)

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

    def calculate_rules_indices(self, rules_shuffled_idx: np.ndarray, fold: int) -> (np.ndarray, np.ndarray):
        """

        :param rules_shuffled_idx:
        :param fold:
        :return:
        """
        test_rules_idx = rules_shuffled_idx[fold::self.denoising_config.crossweight_folds]
        train_rules_idx = [rules_shuffled_idx[x::self.denoising_config.crossweight_folds]
                           for x in range(self.denoising_config.crossweight_folds) if x != fold]
        train_rules_idx = np.concatenate(train_rules_idx, axis=0)
        assert set(test_rules_idx).isdisjoint(set(train_rules_idx))
        return train_rules_idx, test_rules_idx

    def get_cw_samples_labels(self,
                              inputs_x: TensorDataset,
                              rule_matches_z: TensorDataset,
                              labels: np.ndarray,
                              rules_idx: np.ndarray,
                              intersections: np.ndarray = None
                              ) -> (TensorDataset, np.ndarray, np.ndarray):
        """

        :param rules_idx:
        :param inputs_x: encoded samples (samples x features)
        :param labels:
        :param rule_matches_z:  binary matrix that contains info about rules matched in samples (samples x rules)
        :param intersections:
        :return:
        """
        cw_samples = np.zeros((0, inputs_x.shape[1]), dtype="int32")
        cw_labels = np.zeros(0, dtype="int32")
        cw_samples_idx = np.zeros(0, dtype="int32")
        for idx in rules_idx:
            matched_sample_ids = np.where(rule_matches_z[:, idx] == 1)
            if intersections is not None:
                if inputs_x[matched_sample_ids] in intersections:
                    continue
            cw_samples = np.vstack((cw_samples, inputs_x[matched_sample_ids]))
            cw_labels = np.append(cw_labels, labels[matched_sample_ids])
            cw_samples_idx = np.append(cw_samples_idx, matched_sample_ids[0])
        return cw_samples, cw_labels, cw_samples_idx

    def select_samples_n_labels(self,
                                inputs_x: TensorDataset,
                                rule_matches_z: TensorDataset,
                                train_rules_idx:  np.ndarray,
                                test_rules_idx: np.ndarray,
                                labels: np.ndarray
                                ) -> (TensorDataset, np.ndarray, list, TensorDataset, np.ndarray, list):
        """

        :param inputs_x: encoded samples (samples x features)
        :param rule_matches_z:  binary matrix that contains info about rules matched in samples (samples x rules)
        :param train_rules_idx:
        :param test_rules_idx:
        :param labels:
        :return:
        """

        # select train and test samples and labels according to the selected rules idx
        test_samples, test_labels, test_idx = self.get_cw_samples_labels(inputs_x, rule_matches_z, labels,
                                                                         test_rules_idx)
        train_samples, train_labels, train_idx = self.get_cw_samples_labels(inputs_x, rule_matches_z, labels,
                                                                            train_rules_idx, test_samples)

        return train_samples, train_labels, train_idx, test_samples, test_labels, test_idx

    def crossweight_train(self, train_loader: DataLoader):
        """
        Training of crossweight model
        :param train_loader: loader with the data which is used for training (k-1 folds)
        :return:
        """
        self.model.apply(utils.initialize_weights)
        self.model.train()
        for e in range(self.denoising_config.crossweight_epochs):
            for tokens, labels, _ in train_loader:
                tokens, labels = tokens.to(device=device), labels.to(device=device)
                self.denoising_config.optimizer.zero_grad()
                output = self.model(tokens)
                loss = self.denoising_config.criterion(output, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIPPING)
                self.denoising_config.optimizer.step()

    def crossweight_test(self, test_loader: DataLoader, sample_weights: np.ndarray) -> None:
        """
        Testing of trained crossweight model on a hold-out fold and reducing weights of disagreed samples
        :param test_loader: loader with the data which is used for testing (hold-out fold)
        :param sample_weights:
        :return:
        """
        self.model.eval()
        with torch.no_grad():
            for tokens, labels, idx in test_loader:
                tokens, labels, idx = tokens.to(device=device), labels.to(device=device), idx.to(device=device)
                outputs = self.model(tokens)
                _, predicted = torch.max(outputs.data, -1)
                predictions = predicted.tolist()
                for curr_pred in range(len(labels)):
                    gold = labels.tolist()[curr_pred]
                    guess = predictions[curr_pred]
                    curr_id = idx[curr_pred].tolist()
                    if gold != guess:
                        sample_weights[curr_id] *= self.denoising_config.weight_reducing_rate

    def initialise_sample_weights(self, inputs_x: TensorDataset) -> np.ndarray:
        """ Creating an initial sample weights matrix where weights for all samples equal sample_start_weights param"""
        return np.array([self.denoising_config.samples_start_weights] * inputs_x.shape[0], dtype=np.float64)

    def get_cw_data(self,
                    inputs_x: TensorDataset,
                    rule_matches_z: TensorDataset,
                    rules_shuffled_idx: np.ndarray,
                    labels: np.ndarray,
                    fold: int
                    ) -> (DataLoader, DataLoader):
        """
        This function returns train and test loader for crossweight training
        :param inputs_x: encoded samples (samples x features)
        :param rule_matches_z: binary matrix that contains info about rules matched in samples (samples x rules)
        :param rules_shuffled_idx: array of row numbers
        :param labels: binary matrix of labels corresponding to samples (samples x classes)
        :param fold: number of a current hold-out fold
        :return: train and test data as DataLoaders
        """
        train_rules_idx, test_rules_idx = self.calculate_rules_indices(rules_shuffled_idx, fold)
        train_samples, train_labels, train_idx, test_samples, test_labels, test_idx = self.select_samples_n_labels(
            inputs_x, rule_matches_z, train_rules_idx, test_rules_idx, labels)

        test_loader = utils.convert2tensor(test_samples, test_labels, test_idx, self.denoising_config.batch_size)
        train_loader = utils.convert2tensor(train_samples, train_labels, train_idx, self.denoising_config.batch_size)

        self.logger.info("Rules in training set:{}, rules in test set: {}".format(
            len(train_rules_idx), len(test_rules_idx)))

        return train_loader, test_loader

    def get_labels(self, rule_matches_z: TensorDataset, rule_assignments_t: np.ndarray) -> np.ndarray:

        assert rule_matches_z.shape[1] == rule_assignments_t.shape[0], "Check matrices dimensionality!"

        one_hot_labels = rule_matches_z.dot(rule_assignments_t)  # calculate labels
        one_hot_labels[one_hot_labels > 0] = 1
        labels = [np.where(r == 1)[0][0] for r in one_hot_labels]   # todo: decide on multi class labels - now a label is chosen randomly
        return np.stack(labels, axis=0)

    def calculate_sample_weights(self,
                                 inputs_x: TensorDataset,
                                 rule_matches_z: TensorDataset,
                                 rule_assignments_t: np.ndarray,
                                 path_to_weights_dir: str
                                 ) -> None:
        """

        :param inputs_x: encoded samples (samples x features)
        :param rule_matches_z: binary matrix that contains info about rules matched in samples (samples x rules)
        :param rule_assignments_t: binary matrix that contains info about which rule correspond to which label
        :param path_to_weights_dir:
        :return:
        """
        self.logger.info("The trained sample weights will be saved to:{}".format(path_to_weights_dir))

        labels = self.get_labels(rule_matches_z, rule_assignments_t)
        sample_weights = self.initialise_sample_weights(inputs_x)       # initialise sample weights

        for partition in range(self.denoising_config.crossweight_rounds):
            rules_shuffled_idx = utils.get_shuffled_idx(rule_assignments_t)    # shuffle indices anew for each cw round

            for fold in range(self.denoising_config.crossweight_folds):
                train_loader, test_loader = self.get_cw_data(inputs_x, rule_matches_z, rules_shuffled_idx, labels, fold)
                self.crossweight_train(train_loader)
                self.crossweight_test(test_loader, sample_weights)

        utils.save_sample_weights(sample_weights, path_to_weights_dir)

    def denoise(self,
                inputs_x: TensorDataset,
                rule_matches_z: TensorDataset,
                rule_assignments_t: np.ndarray
                ) -> None:
        """
        Denoising of training data labels using CrossWeight method of calculation sample weights
        :param inputs_x: encoded samples (samples x features)
        :param rule_matches_z: binary matrix that contains info about rules matched in samples (samples x rules)
        :param rule_assignments_t: binary matrix that contains info about which rule correspond to which label
        """

        path_to_res_dir = os.path.join(self.denoising_config.path_to_weights, str(date.today()))
        os.makedirs(path_to_res_dir, exist_ok=True)  # Create dir where weights will be saved

        for i in range(self.denoising_config.crossweight_rounds):
            self.logger.info("Crossweight Iteration {} out of {}".format(i, self.denoising_config.crossweight_rounds))

            path_to_weights_dir = os.path.join(path_to_res_dir, str(i))
            os.makedirs(path_to_weights_dir, exist_ok=True)

            self.calculate_sample_weights(inputs_x, rule_matches_z, rule_assignments_t, path_to_weights_dir)

    def train(self,
              inputs_x: TensorDataset,
              rule_matches_z: TensorDataset,
              rule_assignments_t: np.ndarray
              ):
        """

        :param inputs_x: encoded samples (samples x features)
        :param rule_matches_z: binary matrix that contains info about rules matched in samples (samples x rules)
        :param rule_assignments_t: binary matrix that contains info about which rule correspond to which label
        """

        self.denoise(inputs_x, rule_matches_z, rule_assignments_t)
        self.train(inputs_x, rule_matches_z, rule_assignments_t)


if __name__ == '__main__':
    CrossWeightTrainer(BidirectionalLSTM).denoise()


