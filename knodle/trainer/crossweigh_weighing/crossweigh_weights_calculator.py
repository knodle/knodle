import copy
import logging
import os
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader
from joblib import dump
from tqdm import tqdm
from knodle.trainer.ds_model_trainer.ds_model_trainer import DsModelTrainer
from knodle.trainer.config.crossweigh_denoising_config import CrossWeighDenoisingConfig
from knodle.trainer.crossweigh_weighing.utils import set_device, set_seed, check_splitting, return_unique
from knodle.trainer.utils.denoise import get_majority_vote_probs, get_majority_vote_probs_with_no_rel

logger = logging.getLogger(__name__)
torch.set_printoptions(edgeitems=100)
NO_MATCH_CLASS = -1


class CrossWeighWeightsCalculator:
    def __init__(self,
                 model: Module,
                 rule_assignments_t: np.ndarray,
                 inputs_x: TensorDataset,
                 rule_matches_z: np.ndarray,
                 denoising_config: CrossWeighDenoisingConfig = None):

        self.inputs_x = inputs_x
        self.rule_matches_z = rule_matches_z
        self.rule_assignments_t = rule_assignments_t
        self.model = model
        self.model_copy = copy.deepcopy(self.model)

        if denoising_config is None:
            self.denoising_config = CrossWeighDenoisingConfig(self.model)
            logger.info("Default CrossWeigh Config is used: {}".format(self.denoising_config.__dict__))
        else:
            self.denoising_config = denoising_config
            logger.info("Initalized trainer with custom model config: {}".format(self.denoising_config.__dict__))

        self.device = set_device(self.denoising_config.enable_cuda)
        self.sample_weights = self.initialise_sample_weights()

    def calculate_weights(self) -> np.ndarray:
        """
        This function calculates the sample_weights for samples using CrossWeigh method
        :return matrix of the sample sample_weights
        """
        set_seed(self.denoising_config.seed)
        logger.info("======= Denoising with CrossWeigh is started =======")

        os.makedirs(self.denoising_config.path_to_weights, exist_ok=True)

        if self.denoising_config.negative_samples:
            labels = get_majority_vote_probs_with_no_rel(self.rule_matches_z, self.rule_assignments_t, NO_MATCH_CLASS)
        else:
            labels = get_majority_vote_probs(self.rule_matches_z, self.rule_assignments_t)

        for partition in range(self.denoising_config.cw_partitions):
            logger.info("============= CrossWeigh Partition {}/{}: =============".format(partition + 1,
                                                                                         self.denoising_config.cw_partitions))
            rules_shuffled_idx = self._get_shuffled_rules_idx()  # shuffle anew for each cw round
            for fold in range(self.denoising_config.cw_folds):
                self.model = self.model_copy
                train_loader, test_loader = self.get_cw_data(rules_shuffled_idx, labels, fold)
                self.cw_train(train_loader)
                self.cw_test(test_loader)
            logger.info("============ CrossWeigh Partition {} is done ============".format(partition + 1))
        self._save_weights()

        logger.info("======= Denoising with CrossWeigh is completed, sample_weights are saved to {} =======".format(
            self.denoising_config.path_to_weights))
        return self.sample_weights

    def _get_shuffled_rules_idx(self) -> np.ndarray:
        """ Get shuffled row indices of dataset """
        return np.random.rand(self.rule_assignments_t.shape[0]).argsort()

    def initialise_sample_weights(self) -> torch.FloatTensor:
        """ Creates an initial sample sample_weights matrix of size (num_samples x 1) where sample_weights for all samples equal
        sample_start_weights param """
        return torch.FloatTensor([self.denoising_config.samples_start_weights] * self.inputs_x.tensors[0].shape[0])

    def calculate_rules_indices(self, rules_idx: np.ndarray, fold: int) -> (np.ndarray, np.ndarray):
        """
        Calculates the indices of the samples which are to be included in CrossWeigh training and test sets
        :param rules_idx: all rules indices (shuffled) that are to be splitted into cw training & cw test set rules
        :param fold: number of a current hold-out fold
        :return: two arrays containing indices of rules that will be used for cw training and cw test set accordingly
        """
        test_rules_idx = rules_idx[fold::self.denoising_config.cw_folds]
        train_rules_idx = [rules_idx[x::self.denoising_config.cw_folds] for x in range(self.denoising_config.cw_folds)
                           if x != fold]
        train_rules_idx = np.concatenate(train_rules_idx, axis=0)

        if not set(test_rules_idx).isdisjoint(set(train_rules_idx)):
            raise ValueError("Splitting into train and test rules is done incorrectly.")

        return train_rules_idx, test_rules_idx

    def get_cw_data(self, rules_idx: np.ndarray, labels: np.ndarray, fold: int) -> (DataLoader, DataLoader):
        """
        This function returns train and test dataloaders for CrossWeigh training. Each dataloader comprises encoded
        samples, labels and sample indices in the original matrices
        :param rules_idx: shuffled rules indices
        :param labels: labels of all training samples
        :param fold: number of a current hold-out fold
        :return: dataloaders for cw training and testing
        """
        train_rules_idx, test_rules_idx = self.calculate_rules_indices(rules_idx, fold)

        # select train and test samples and labels according to the selected rules idx
        test_samples, test_labels, test_idx = self._get_cw_samples_labels_idx(labels, test_rules_idx)
        train_samples, train_labels, train_idx = self._get_cw_samples_labels_idx(labels, train_rules_idx, test_idx)

        # debug: check that splitting was done correctly
        check_splitting(train_samples, train_labels, train_idx, self.inputs_x.tensors[0], labels)
        check_splitting(test_samples, test_labels, test_idx, self.inputs_x.tensors[0], labels)

        test_loader = self.cw_convert2tensor(test_samples, test_labels, test_idx)
        train_loader = self.cw_convert2tensor(train_samples, train_labels, train_idx)

        logger.info("\nFold {}          Rules in training set:{}, rules in test set: {}, samples in training set: {}, "
                    "samples in test set: {}".format(fold, len(train_rules_idx), len(test_rules_idx),
                                                     len(train_samples), len(test_samples)))
        return train_loader, test_loader

    def _get_cw_samples_labels_idx(
            self, labels: np.ndarray, indices: np.ndarray, check_intersections: np.ndarray = None
    ) -> (torch.Tensor, np.ndarray, np.ndarray):
        """
        Extracts the samples and labels from the original matrices by the indices. If intersection is filled with
        another sample matrix, it also checks whether the sample is not in this other matrix yet.
        :param labels: all training samples labels, shape=(num_samples, num_classes)
        :param indices: indices of the samples & labels which are to be included in the set
        :param check_intersections: optional parameter that indicates that intersections should be checked (used to
        exclude the sentences from the CrossWeigh training set which are already in CrossWeigh test set)
        :return: samples, labels and indices in the original matrix
        """
        cw_samples = torch.LongTensor(0, self.inputs_x.tensors[0].shape[1])
        cw_labels = np.zeros((0, self.rule_assignments_t.shape[1]), dtype="int64")
        cw_samples_idx = np.zeros(0, dtype="int64")
        for idx in indices:
            matched_sample_ids = np.where(self.rule_matches_z[:, idx] == 1)[0]
            if len(matched_sample_ids) == 0:  # if nothing matched this rule
                continue
            matched_sample_ids = return_unique(matched_sample_ids, cw_samples_idx)
            if check_intersections is not None:
                matched_sample_ids = return_unique(matched_sample_ids, check_intersections)
            cw_samples = torch.cat((cw_samples, self.inputs_x.tensors[0][matched_sample_ids]), 0)
            cw_labels = np.append(cw_labels, labels[matched_sample_ids, :], 0)
            cw_samples_idx = np.append(cw_samples_idx, matched_sample_ids)
        return cw_samples, cw_labels, cw_samples_idx

    def cw_convert2tensor(
            self, samples: torch.Tensor, labels: np.ndarray, idx: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        """
        Turns the input data (encoded samples, encoded labels, indices in the original matrices) to a DataLoader
        which could be used for further model training or testing
        """
        tensor_words = samples.to(device=self.device)
        tensor_target = torch.LongTensor(labels).to(device=self.device)
        tensor_idx = torch.LongTensor(idx).to(device=self.device)

        dataset = torch.utils.data.TensorDataset(tensor_words, tensor_target, tensor_idx)
        return torch.utils.data.DataLoader(dataset, batch_size=self.denoising_config.batch_size, shuffle=shuffle)

    def cw_train(self, train_loader: DataLoader):
        """
        Training of CrossWeigh model
        :param train_loader: loader with the data which is used for training (k-1 folds)
        """
        self.model.train()
        for _ in tqdm(range(self.denoising_config.cw_epochs)):
            for tokens, labels, _ in train_loader:
                self.denoising_config.criterion.weight = self.denoising_config.class_weights
                self.denoising_config.criterion.reduction = "none"
                self.denoising_config.optimizer.zero_grad()
                output = self.model(tokens)
                loss = self._get_train_loss(self.denoising_config.criterion, output, labels, self.sample_weights)
                # loss = self.denoising_config.criterion(output, labels, weight=self.denoising_config.class_weights)
                loss.backward()
                if self.denoising_config.use_grad_clipping:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.denoising_config.grad_clipping)
                self.denoising_config.optimizer.step()

    def _get_train_loss(self, criterion, output, labels, weights):
        return (criterion(output, labels) * weights).sum() / self.denoising_config.class_weights[labels].sum()

    def cw_test(self, test_loader: DataLoader) -> None:
        """
        This function tests of trained CrossWeigh model on a hold-out fold, compared the predicted labels with the
        ones got with weak supervision and reduces sample_weights of disagreed samples
        :param test_loader: loader with the data which is used for testing (hold-out fold)
        """
        self.model.eval()
        correct_predictions, wrong_predictions = 0, 0
        with torch.no_grad():
            for tokens, labels, idx in test_loader:
                outputs = self.model(tokens)
                _, predicted = torch.max(outputs.data, -1)
                predictions = predicted.tolist()
                for curr_pred in range(len(predictions)):
                    gold = labels.tolist()[curr_pred]
                    gold_classes = [idx for idx, value in enumerate(gold) if value > 0]
                    guess = predictions[curr_pred]
                    if guess not in gold_classes:
                        correct_predictions += 1
                        curr_id = idx[curr_pred].tolist()
                        self.sample_weights[curr_id] *= self.denoising_config.weight_reducing_rate
                    else:
                        wrong_predictions += 1
        logger.info("Correct predictions: {:.3f}%, wrong predictions: {:.3f}%".format(
            correct_predictions * 100/(correct_predictions+wrong_predictions),
            wrong_predictions * 100/(correct_predictions+wrong_predictions)))

    def _save_weights(self) -> None:
        # tokens_with_weights = np.hstack((self.inputs_x.tensors[0], self.sample_weights[:, None]))
        dump(self.sample_weights, self.denoising_config.path_to_weights + "/" + "sample_weights")


