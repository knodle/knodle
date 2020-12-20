import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader

from knodle.trainer.crossweight_denoising import utils
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


class CrossWeightWeightsCalculator:

    def __init__(self,
                 model: Module,
                 rule_assignments_t: np.ndarray,
                 inputs_x: TensorDataset,
                 rule_matches_z: np.ndarray,
                 denoising_config: CrossWeightDenoisingConfig = None):

        self.inputs_x = inputs_x
        self.rule_matches_z = rule_matches_z
        self.rule_assignments_t = rule_assignments_t
        self.model = model

        self.logger = logging.getLogger(__name__)

        if denoising_config is None:
            self.dncnfg = CrossWeightDenoisingConfig(self.model)
            self.logger.info(
                "Default CrossWeight Config is used: {}".format(self.dncnfg.__dict__)
            )
        else:
            self.dncnfg = denoising_config
            self.logger.info(
                "Initalized trainer with custom model config: {}".format(
                    self.dncnfg.__dict__
                )
            )

    def calculate_weights(self) -> np.ndarray:
        """
        This function calculates the weights for samples using CrossWeight method
        :return matrix of the sample weights
        """

        self.logger.info("======= Denoising with CrossWeight is started =======")

        os.makedirs(self.dncnfg.path_to_weights, exist_ok=True)
        labels = utils.get_labels(self.rule_matches_z, self.rule_assignments_t)
        sample_weights = self.initialise_sample_weights()

        for i in range(self.dncnfg.cw_partitions):

            self.logger.info("CrossWeight Partition {} out of {}:".format(i+1, self.dncnfg.cw_partitions))
            rules_shuffled_idx = self.get_shuffled_idx()        # shuffle anew for each cw round

            for fold in range(self.dncnfg.cw_folds):

                train_loader, test_loader = self.get_cw_data(rules_shuffled_idx, labels, fold)
                self.cw_train(train_loader)
                self.cw_test(test_loader, sample_weights)
            self.logger.info("CrossWeight Partition {} is done".format(i + 1))

        self.save_weights(sample_weights)
        self.logger.info("======= Denoising with CrossWeight is completed, weights are saved to {} =======".format(
            self.dncnfg.path_to_weights))

        return sample_weights

    def get_shuffled_idx(self) -> np.ndarray:
        """ Get shuffled row indices of dataset """
        return np.random.rand(self.rule_assignments_t.shape[0]).argsort()

    def save_weights(self, weights: np.ndarray) -> None:
        tokens_with_weights = np.hstack((self.inputs_x.tensors[0], weights[:, None]))
        np.save(self.dncnfg.path_to_weights + "/" + "sample_weights", tokens_with_weights)

    def cw_train(self, train_loader: DataLoader):
        """
        Training of crossweight model
        :param train_loader: loader with the data which is used for training (k-1 folds)
        """
        self.model.train()
        for e in range(self.dncnfg.cw_epochs):
            for tokens, labels, _ in train_loader:
                tokens, labels = tokens.to(device=device), labels.to(device=device)
                self.dncnfg.optimizer.zero_grad()
                output = self.model(tokens)
                loss = self.dncnfg.criterion(output, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIPPING)
                self.dncnfg.optimizer.step()

    def cw_test(self, test_loader: DataLoader, sample_weights: np.ndarray) -> None:
        """
        This function tests of trained crossweight model on a hold-out fold, compared the predicted labels with the
        ones got with weak supervision and reduces weights of disagreed samples
        :param test_loader: loader with the data which is used for testing (hold-out fold)
        :param sample_weights: current sample weights as they have been calculated in the cw rounds before
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
                        sample_weights[curr_id] *= self.dncnfg.weight_reducing_rate

    def initialise_sample_weights(self) -> np.ndarray:
        """ Creates an initial sample weights matrix of size (num_samples x 1) where weights for all samples equal
        sample_start_weights param """
        return np.array([self.dncnfg.samples_start_weights] * self.inputs_x.tensors[0].shape[0], dtype=np.float64)

    def calculate_rules_indices(self, rules_idx: np.ndarray, fold: int) -> (np.ndarray, np.ndarray):
        """
        Calculates the indices of the samples which are to be included in crossweight training and test sets
        :param rules_idx: all rules indices (shuffled) that are to be splitted into cw training & cw test set rules
        :param fold: number of a current hold-out fold
        :return: two arrays containing indices of rules that will be used for cw training and cw test set correspondingly
        """
        test_rules_idx = rules_idx[fold::self.dncnfg.cw_folds]
        train_rules_idx = [rules_idx[x::self.dncnfg.cw_folds] for x in range(self.dncnfg.cw_folds) if x != fold]
        train_rules_idx = np.concatenate(train_rules_idx, axis=0)
        assert set(test_rules_idx).isdisjoint(set(train_rules_idx))
        return train_rules_idx, test_rules_idx

    def get_cw_data(self, rules_idx: np.ndarray, labels: np.ndarray, fold: int) -> (DataLoader, DataLoader):
        """
        This function returns train and test dataloaders for crossweight training. Each dataloader comprises encoded
        samples, labels and sample indices in the original matrices
        :param rules_idx: shuffled rules indices
        :param labels: labels of all training samples
        :param fold: number of a current hold-out fold
        :return: dataloaders for cw training and testing
        """

        train_rules_idx, test_rules_idx = self.calculate_rules_indices(rules_idx, fold)

        # select train and test samples and labels according to the selected rules idx
        test_samples, test_labels, test_idx = self.get_cw_samples_labels_idx(labels, test_rules_idx)
        train_samples, train_labels, train_idx = self.get_cw_samples_labels_idx(labels, train_rules_idx, test_samples)

        test_loader = self.cw_convert2tensor(test_samples, test_labels, test_idx)
        train_loader = self.cw_convert2tensor(train_samples, train_labels, train_idx)

        self.logger.info("Fold {}       Rules in training set:{}, samples in training set:{}, rules in test set: {}, "
                         "samples in test set: {}".format(fold, len(train_rules_idx), len(train_samples),
                                                          len(test_rules_idx), len(test_samples)))

        return train_loader, test_loader

    def get_cw_samples_labels_idx(
            self, labels: np.ndarray, idx: np.ndarray, check_intersections: np.ndarray = None
    ) -> (TensorDataset, np.ndarray, np.ndarray):
        """
        Extracts the samples and labels from the original matrices by the indices. If intersection is filled with
        another sample matrix, it also checks whether the sample is not in this other matrix yet.
        :param labels: all sample labels
        :param idx: indices of the samples & labels which are to be included in the set
        :param check_intersections: optional parameter that indicates that intersections should be checked (used to
        exclude the sentences from the crossweight training set which are already in crossweight test set)
        :return: samples, labels and indices in the original matrix
        """
        cw_samples = np.zeros((0, self.inputs_x.tensors[0].shape[1]), dtype="int32")
        cw_labels = np.zeros(0, dtype="int32")
        cw_samples_idx = np.zeros(0, dtype="int32")
        for idx in idx:
            matched_sample_ids = np.where(self.rule_matches_z[:, idx] == 1)
            if check_intersections is not None:
                if self.inputs_x[matched_sample_ids] in check_intersections:
                    continue
            cw_samples = np.vstack((cw_samples, self.inputs_x.tensors[0][matched_sample_ids]))
            cw_labels = np.append(cw_labels, labels[matched_sample_ids])
            cw_samples_idx = np.append(cw_samples_idx, matched_sample_ids[0])
        return cw_samples, cw_labels, cw_samples_idx

    def cw_convert2tensor(
            self, samples: np.ndarray, labels: np.ndarray, idx: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        """
        Turns the input data (encoded samples, encoded labels, indices in the original matrices) to a DataLoader
        which could be used for further model training or testing
        """

        tensor_words = torch.LongTensor(samples).to(device=device)
        tensor_target = torch.LongTensor(labels).to(device=device)
        tensor_idx = torch.LongTensor(idx).to(device=device)

        dataset = torch.utils.data.TensorDataset(tensor_words, tensor_target, tensor_idx)
        return torch.utils.data.DataLoader(dataset, batch_size=self.dncnfg.batch_size, shuffle=shuffle)
