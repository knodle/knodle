import copy
import logging
import os
import random
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import TensorDataset, DataLoader
from joblib import dump

from knodle.trainer.baseline.majority import MajorityVoteTrainer
from knodle.trainer.wscrossweigh.utils import check_splitting, return_unique
from knodle.trainer.utils import log_section
from knodle.transformation.filter import filter_empty_probabilities
from knodle.transformation.majority import z_t_matrices_to_majority_vote_probs
from knodle.transformation.torch_input import input_info_labels_to_tensordataset, input_labels_to_tensordataset

logger = logging.getLogger(__name__)
torch.set_printoptions(edgeitems=100)


class WSCrossWeighWeightsCalculator(MajorityVoteTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wscrossweigh_model = copy.deepcopy(self.model)
        self.sample_weights = torch.empty(0)

    def calculate_weights(self) -> torch.FloatTensor:
        """
        This function calculates the sample_weights for samples using WSCrossWeigh method
        :return matrix of the sample sample_weights
        """

        if self.trainer_config.folds < 2:
            raise ValueError("Number of folds should be at least 2 to perform WSCrossWeigh denoising")

        logger.info("======= Denoising with WSCrossWeigh is started =======")
        os.makedirs(self.trainer_config.caching_folder, exist_ok=True)

        labels = z_t_matrices_to_majority_vote_probs(
            self.rule_matches_z, self.mapping_rules_labels_t, self.trainer_config.other_class_id
        )

        if self.trainer_config.filter_non_labelled:
            self.model_input_x, labels, self.rule_matches_z = filter_empty_probabilities(
                self.model_input_x, labels, self.rule_matches_z
            )

        other_sample_ids = self._get_other_sample_ids(labels) if self.trainer_config.other_class_id else None
        rules_samples_ids_dict = self._get_rules_samples_ids_dict()

        self.sample_weights = self.initialise_sample_weights()

        for partition in range(self.trainer_config.partitions):
            log_section(f"CrossWeigh Partition {partition + 1}/{self.trainer_config.partitions}:", logger)

            shuffled_rules_ids = self._get_shuffled_rules_idx()  # shuffle anew for each cw round

            for fold in range(self.trainer_config.folds):
                # for each fold the model is trained from scratch
                self.wscrossweigh_model = copy.deepcopy(self.model).to(self.trainer_config.device)
                train_loader, test_loader = self.get_cw_data(
                    shuffled_rules_ids, rules_samples_ids_dict, labels, fold, other_sample_ids
                )
                self._train_loop(train_loader)
                self.cw_test(test_loader)

            log_section(f"CrossWeigh Partition {partition + 1} is done", logger)

        dump(
            self.sample_weights, os.path.join(self.trainer_config.caching_folder,
                                              f"sample_weights_{self.trainer_config.caching_suffix}.lib")
             )
        logger.info("======= Denoising with WSCrossWeigh is completed =======")
        return self.sample_weights

    def _get_other_sample_ids(self, labels: np.ndarray) -> List[int]:
        return np.where(labels[:, self.trainer_config.other_class_id] == 1)[0].tolist()

    def _get_shuffled_rules_idx(self) -> List[int]:
        """ Get shuffled row indices of dataset """
        rel_rules_ids = [rule_idx for rule_idx in range(0, self.mapping_rules_labels_t.shape[0])]
        random.shuffle(rel_rules_ids)
        return rel_rules_ids

    def _get_rules_samples_ids_dict(self):
        """
        This function creates a dictionary {rule id : sample id where this rule matched}. The dictionary is needed as a
        support tool for faster calculation of cw train and cw test sets
         """
        if isinstance(self.rule_matches_z, sp.csr_matrix):
            rules_samples_ids_dict = {key: [] for key in range(self.rule_matches_z.shape[1])}
            for row, col in zip(*self.rule_matches_z.nonzero()):
                rules_samples_ids_dict[col].append(row)
        else:
            rules_samples_ids_dict = dict((i, set()) for i in range(0, self.rule_matches_z.shape[1]))
            for row_idx, row in enumerate(self.rule_matches_z):
                rules = np.where(row == 1)[0].tolist()
                for rule in rules:
                    rules_samples_ids_dict[rule].add(row_idx)
        return rules_samples_ids_dict

    def initialise_sample_weights(self) -> torch.FloatTensor:
        """ Initialise a sample_weights matrix (num_samples x 1): weights for all samples equal sample_start_weights """
        return torch.FloatTensor([self.trainer_config.samples_start_weights] * self.model_input_x.tensors[0].shape[0])

    def calculate_rules_indices(self, rules_idx: list, fold: int) -> (np.ndarray, np.ndarray):
        """
        Calculates the indices of the samples which are to be included in WSCrossWeigh training and test sets
        :param rules_idx: all rules indices (shuffled) that are to be splitted into cw training & cw test set rules
        :param fold: number of a current hold-out fold
        :return: two arrays containing indices of rules that will be used for cw training and cw test set accordingly
        """
        test_rules_idx = rules_idx[fold::self.trainer_config.folds]
        train_rules_idx = [rules_idx[x::self.trainer_config.folds] for x in range(self.trainer_config.folds)
                           if x != fold]
        train_rules_idx = [ids for sublist in train_rules_idx for ids in sublist]

        if not set(test_rules_idx).isdisjoint(set(train_rules_idx)):
            raise ValueError("Splitting into train and test rules is done incorrectly.")

        return train_rules_idx, test_rules_idx

    def get_cw_data(
            self, rules_ids: List[int], rules_samples_ids_dict: Dict, labels: np.ndarray, fold: int,
            other_sample_ids: List[int]
    ) -> (DataLoader, DataLoader):
        """
        This function returns train and test dataloaders for WSCrossWeigh training. Each dataloader comprises encoded
        samples, labels and sample indices in the original matrices
        :param rules_ids: shuffled rules indices
        :param labels: labels of all training samples
        :param fold: number of a current hold-out fold
        :return: dataloaders for cw training and testing
        """
        train_rules_idx, test_rules_idx = self.calculate_rules_indices(rules_ids, fold)

        # select train and test samples and labels according to the selected rules idx
        test_samples, test_labels, test_idx = self._get_cw_samples_labels_idx(
            labels, test_rules_idx, rules_samples_ids_dict
        )
        test_loader = self._make_dataloader(
            input_info_labels_to_tensordataset(test_samples, test_idx, test_labels)
        )

        train_samples, train_labels, _ = self._get_cw_samples_labels_idx(
            labels, train_rules_idx, rules_samples_ids_dict, test_idx, other_sample_ids
        )
        train_loader = self._make_dataloader(
            input_labels_to_tensordataset(train_samples, train_labels)
        )

        logger.info(f"Fold {fold}/{self.trainer_config.folds}  Rules in training set: {len(train_rules_idx)}, "
                    f"rules in test set: {len(test_rules_idx)}, samples in training set: {len(train_samples)}, "
                    f"samples in test set: {len(test_samples)}")
        return train_loader, test_loader

    def _get_cw_samples_labels_idx(
            self, labels: np.ndarray, indices: list, rules_samples_ids_dict: Dict, check_intersections: np.ndarray = None,
            other_sample_ids: list = None
    ) -> (torch.Tensor, np.ndarray, np.ndarray):
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

        if other_sample_ids is not None:
            sample_ids = list(set(sample_ids).union(set(other_sample_ids)))

        if check_intersections is not None:
            sample_ids = return_unique(np.array(sample_ids), check_intersections)
        cw_samples_dataset = TensorDataset(torch.Tensor(self.model_input_x.tensors[0][sample_ids]))
        cw_labels = np.array(labels[sample_ids])
        cw_samples_idx = np.array(sample_ids)

        check_splitting(cw_samples_dataset, cw_labels, cw_samples_idx, self.model_input_x.tensors[0], labels)
        return cw_samples_dataset, cw_labels, cw_samples_idx

    def cw_test(self, test_loader: DataLoader) -> None:
        """
        This function tests of trained WSCrossWeigh model on a hold-out fold, compared the predicted labels with the
        ones got with weak supervision and reduces sample_weights of disagreed samples
        :param test_loader: loader with the data which is used for testing (hold-out fold)
        """
        self.wscrossweigh_model.eval()
        correct_predictions, wrong_predictions = 0, 0

        with torch.no_grad():
            for batch in test_loader:
                features, labels = self._load_batch(batch)
                data_features, data_indices = features[:-1], features[-1]

                outputs = self.wscrossweigh_model(*data_features)
                outputs = outputs[0] if not isinstance(outputs, torch.Tensor) else outputs
                _, predicted = torch.max(outputs.data, -1)
                predictions = predicted.tolist()

                for curr_pred in range(len(predictions)):
                    gold = labels.tolist()[curr_pred]
                    gold_classes = gold.index(max(gold))
                    guess = predictions[curr_pred]
                    if guess != gold_classes:       # todo: what if more than one class could be predicted? e.g. conll
                        wrong_predictions += 1
                        curr_id = data_indices[curr_pred].tolist()
                        self.sample_weights[curr_id] *= self.trainer_config.weight_reducing_rate
                    else:
                        correct_predictions += 1

        logger.info("Correct predictions: {:.3f}%, wrong predictions: {:.3f}%".format(
            correct_predictions * 100 / (correct_predictions + wrong_predictions),
            wrong_predictions * 100 / (correct_predictions + wrong_predictions)))
