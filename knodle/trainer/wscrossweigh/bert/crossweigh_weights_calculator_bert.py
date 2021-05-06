import copy
import logging
import os
import random
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader
from joblib import dump
from tqdm import tqdm
from knodle.trainer.wscrossweigh.config import WSCrossWeighDenoisingConfig
from knodle.trainer.wscrossweigh.utils import set_seed, return_unique
from knodle.trainer.utils import log_section

logger = logging.getLogger(__name__)
torch.set_printoptions(edgeitems=100)
# NO_RELATION_CLASS = 41


class WSCrossWeighWeightsCalculator:

    def __init__(
            self,
            model: Module,
            rule_assignments_t: np.ndarray,
            inputs_x: TensorDataset,
            rule_matches_z: np.ndarray,
            output_dir: str,
            denoising_config: WSCrossWeighDenoisingConfig = None,
            other_class_id: int = None):

        self.inputs_x = inputs_x
        self.rule_matches_z = rule_matches_z
        self.rule_assignments_t = rule_assignments_t
        self.model = model
        self.wscrossweigh_model = copy.deepcopy(self.model)
        self.output_dir = output_dir
        self.no_relation_class = other_class_id

        if denoising_config is None:
            self.denoising_config = WSCrossWeighDenoisingConfig(self.model)
            logger.info(f"Default WSCrossWeigh Config is used: {self.denoising_config.__dict__}")
        else:
            self.denoising_config = denoising_config
            logger.info(f"Initalized trainer with custom model config: {self.denoising_config.__dict__}")

        self.sample_weights = self.initialise_sample_weights()

    def calculate_weights(self) -> torch.FloatTensor:
        """
        This function calculates the sample_weights for samples using WSCrossWeigh method
        :return matrix of the sample sample_weights
        """
        set_seed(self.denoising_config.seed)

        if self.denoising_config.cw_folds < 2:
            raise ValueError("Number of folds should be at least 2 to perform WSCrossWeigh denoising")

        logger.info("======= Denoising with WSCrossWeigh is started =======")
        os.makedirs(self.output_dir, exist_ok=True)

        labels = get_labels(self.rule_matches_z, self.rule_assignments_t, self.denoising_config.other_class_id)
        rules_samples_ids_dict = self._get_rules_samples_ids_dict()

        for partition in range(self.denoising_config.cw_partitions):
            log_section(f"WSCrossWeigh Partition {partition + 1}/{self.denoising_config.cw_partitions}:", logger)

            shuffled_rules_ids, no_match_ids = self._get_shuffled_rules_idx()  # shuffle anew for each cw round

            for fold in range(self.denoising_config.cw_folds):
                # for each fold the model is trained from scratch
                self.wscrossweigh_model = copy.deepcopy(self.model).to(self.denoising_config.device)
                train_loader, test_loader = self.get_cw_data(
                    shuffled_rules_ids, no_match_ids, rules_samples_ids_dict, labels, fold
                )
                self.cw_train(train_loader)
                self.cw_test(test_loader)

            log_section(f"WSCrossWeigh Partition {partition + 1} is done", logger)

        dump(self.sample_weights, os.path.join(self.output_dir, "sample_weights.lib"))
        logger.info("======= Denoising with WSCrossWeigh is completed =======")
        return self.sample_weights

    def _get_shuffled_rules_idx(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Get shuffled row indices of dataset """
        no_rel_rules_ids = np.where(self.rule_assignments_t[:, self.no_relation_class] == 1)[0].tolist()
        rel_rules_ids = [rule_idx for rule_idx in range(0, self.rule_assignments_t.shape[0])
                         if rule_idx not in no_rel_rules_ids]
        random.shuffle(rel_rules_ids)
        return rel_rules_ids, no_rel_rules_ids

    def _get_rules_samples_ids_dict(self):
        """
        This function creates a dictionary {rule id : sample id where this rule matched}. The dictionary is needed as a
        support tool for faster calculation of cw train and cw test sets
         """
        rules_samples_ids_dict = dict((i, set()) for i in range(0, self.rule_matches_z.shape[1]))
        for row_idx, row in enumerate(self.rule_matches_z):
            rules = np.where(row == 1)[0].tolist()
            for rule in rules:
                rules_samples_ids_dict[rule].add(row_idx)
        return rules_samples_ids_dict

    def initialise_sample_weights(self) -> torch.FloatTensor:
        """
        Creates an initial sample sample_weights matrix of size (num_samples x 1) where sample_weights for all
        samples equal sample_start_weights param
        """
        return torch.FloatTensor([self.denoising_config.samples_start_weights] * self.inputs_x.tensors[0].shape[0])

    def calculate_rules_indices(
            self, rules_idx: list, no_match_rule_ids: list, fold: int) -> (np.ndarray, np.ndarray):
        """
        Calculates the indices of the samples which are to be included in WSCrossWeigh training and test sets
        :param no_match_rule_ids:
        :param rules_idx: all rules indices (shuffled) that are to be splitted into cw training & cw test set rules
        :param fold: number of a current hold-out fold
        :return: two arrays containing indices of rules that will be used for cw training and cw test set accordingly
        """
        test_rules_idx = rules_idx[fold::self.denoising_config.cw_folds]
        train_rules_idx = [rules_idx[x::self.denoising_config.cw_folds] for x in range(self.denoising_config.cw_folds)
                           if x != fold]
        all_train_rules_idx = [ids for sublist in train_rules_idx for ids in sublist] + no_match_rule_ids

        if not set(test_rules_idx).isdisjoint(set(all_train_rules_idx)):
            raise ValueError("Splitting into train and test rules is done incorrectly.")

        return all_train_rules_idx, test_rules_idx

    def get_cw_data(
            self, rules_ids: np.ndarray, no_match_rule_ids: np.ndarray, rules_samples_ids_dict: dict,
            labels: np.ndarray, fold: int
    ) -> (DataLoader, DataLoader):
        """
        This function returns train and test dataloaders for WSCrossWeigh training. Each dataloader comprises encoded
        samples, labels and sample indices in the original matrices
        :param rules_samples_ids_dict:
        :param no_match_rule_ids:
        :param rules_ids:
        :param rules_idx: shuffled rules indices
        :param labels: labels of all training samples
        :param fold: number of a current hold-out fold
        :return: dataloaders for cw training and testing
        """
        train_rules_idx, test_rules_idx = self.calculate_rules_indices(rules_ids, no_match_rule_ids, fold)

        # select train and test samples and labels according to the selected rules idx
        test_samples, test_labels, test_idx = self._get_cw_samples_labels_idx(
            labels, test_rules_idx, rules_samples_ids_dict
        )

        train_samples, train_labels, train_idx = self._get_cw_samples_labels_idx(
            labels, train_rules_idx, rules_samples_ids_dict, test_idx
        )
        # debug: check that splitting was done correctly
        # check_splitting(train_samples, train_labels, train_idx, self.inputs_x.tensors[0], labels)
        # check_splitting(test_samples, test_labels, test_idx, self.inputs_x.tensors[0], labels)

        # test_loader = build_bert_features_labels_ids_dataloader(
        #     test_samples, test_idx, test_labels, self.denoising_config.batch_size
        # )

        test_loader = self._make_dataloader(
            *test_samples.tensors, torch.Tensor(test_idx).long(), torch.Tensor(test_labels).float()
        )

        # train_loader = build_bert_feature_labels_dataloader(
        #     train_samples, train_labels, self.denoising_config.batch_size
        # )
        train_loader = self._make_dataloader(TensorDataset(*train_samples.tensors, train_labels))

        logger.info(f"Fold {fold}/{self.denoising_config.cw_folds}  Rules in training set: {len(train_rules_idx)}, "
                    f"rules in test set: {len(test_rules_idx)}, samples in training set: {len(train_samples)}, "
                    f"samples in test set: {len(test_samples)}")
        return train_loader, test_loader

    def _get_cw_samples_labels_idx(
            self, labels: np.ndarray, indices: list, rules_samples_ids_dict: Dict, check_intersections: np.ndarray = None,
    ) -> (TensorDataset, np.ndarray, np.ndarray):
        """
        Extracts the samples and labels from the original matrices by indices. If intersection is filled with
        another sample matrix, it also checks whether the sample is not in this other matrix yet.
        :param labels: all training samples labels, shape=(num_samples, num_classes)
        :param indices: indices of rules; samples, where these rules matched & their labels are to be included in set
        :param rules_samples_ids_dict: dictionary {rule_id : sample_ids}
        :param check_intersections: optional parameter that indicates that intersections should be checked (used to
        exclude the sentences from the WSCrossWeigh training set which are already in WSCrossWeigh test set)
        :return: samples, labels and indices in the original matrix
        """
        sample_ids = [list(rules_samples_ids_dict.get(idx)) for idx in indices]
        sample_ids = list(set([value for sublist in sample_ids for value in sublist]))
        if check_intersections is not None:
            sample_ids = return_unique(np.array(sample_ids), check_intersections)
        cw_samples_dataset = TensorDataset(
            torch.LongTensor(self.inputs_x.tensors[0][sample_ids]),
            torch.LongTensor(self.inputs_x.tensors[1][sample_ids]))
        cw_labels = np.array(labels[sample_ids])
        cw_samples_idx = np.array(sample_ids)
        return cw_samples_dataset, cw_labels, cw_samples_idx

    def cw_train(self, train_loader: DataLoader):
        """
        Training of WSCrossWeigh model
        :param train_loader: loader with the data which is used for training (k-1 folds)
        """
        self.wscrossweigh_model.train()
        for curr_epoch in range(self.denoising_config.cw_epochs):
            logger.info("Epoch: {}".format(curr_epoch))
            for input_ids_batch, attention_mask_batch, labels in tqdm(train_loader):

                inputs = {
                    "input_ids": input_ids_batch.to(self.denoising_config.device),
                    "attention_mask": attention_mask_batch.to(self.denoising_config.device)
                }
                labels = labels.to(self.denoising_config.device)

                self.model.zero_grad()
                self.denoising_config.optimizer.zero_grad()

                predictions = self.wscrossweigh_model(**inputs)
                loss = self.denoising_config.criterion(predictions[0], labels, weight=self.denoising_config.class_weights)

                loss.backward()
                if isinstance(self.denoising_config.grad_clipping, int):
                    nn.utils.clip_grad_norm_(self.wscrossweigh_model.parameters(), self.denoising_config.grad_clipping)
                self.denoising_config.optimizer.step()

    def cw_test(self, test_loader: DataLoader) -> None:
        """
        This function tests of trained WSCrossWeigh model on a hold-out fold, compared the predicted labels with the
        ones got with weak supervision and reduces sample_weights of disagreed samples
        :param test_loader: loader with the data which is used for testing (hold-out fold)
        """
        self.wscrossweigh_model.eval()
        correct_predictions, wrong_predictions = 0, 0

        with torch.no_grad():
            for tokens, idx, labels in test_loader:
                outputs = self.wscrossweigh_model(tokens)
                _, predicted = torch.max(outputs.data, -1)
                predictions = predicted.tolist()
                for curr_pred in range(len(predictions)):
                    gold = labels.tolist()[curr_pred]
                    gold_classes = gold.index(max(gold))
                    guess = predictions[curr_pred]
                    if guess != gold_classes:
                    # if guess not in gold_classes:
                        wrong_predictions += 1
                        curr_id = idx[curr_pred].tolist()
                        self.sample_weights[curr_id] *= self.denoising_config.weight_reducing_rate
                    else:
                        correct_predictions += 1

        logger.info("Correct predictions: {:.3f}%, wrong predictions: {:.3f}%".format(
            correct_predictions * 100/(correct_predictions+wrong_predictions),
            wrong_predictions * 100/(correct_predictions+wrong_predictions)))