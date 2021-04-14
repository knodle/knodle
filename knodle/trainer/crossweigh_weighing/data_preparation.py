import logging
import random
from typing import List, Dict

import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset

from knodle.trainer.crossweigh_weighing.utils import return_unique, check_splitting
from knodle.transformation.torch_input import input_info_labels_to_tensordataset, input_labels_to_tensordataset

logger = logging.getLogger(__name__)

# todo: check the tests!


def k_folds_splitting(
        data_features: np.ndarray, labels: np.ndarray, rule_matches_z: np.ndarray, partitions: int, folds: int,
        other_class_id: int = None
):
    # todo: here and further: rename data_features!
    train_datasets, test_datasets = [], []
    other_sample_ids = get_other_sample_ids(labels, other_class_id) if other_class_id else None
    rules_samples_ids_dict = get_rules_samples_ids_dict(rule_matches_z)
    rel_rules_ids = [rule_idx for rule_idx in range(0, rule_matches_z.shape[1])]

    for partition in range(partitions):
        logger.info(f"CrossWeigh Partition {partition + 1}/{partitions}:")
        random.shuffle(rel_rules_ids)      # shuffle anew for each cw round
        for fold in range(folds):
            train_dataset, test_dataset = get_train_test_datasets_by_rule_indices(
                data_features, rel_rules_ids, rules_samples_ids_dict, labels, fold, folds, other_sample_ids
            )
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)

    return train_datasets, test_datasets


def get_other_sample_ids(labels: np.ndarray, other_class_id: int) -> List[int]:
    """ Returns ids of all samples that belong to the 'other_class' """
    return np.where(labels[:, other_class_id] == 1)[0].tolist()


def get_rules_samples_ids_dict(rule_matches_z):
    """
    This function creates a dictionary {rule id : sample id where this rule matched}. The dictionary is needed as a
    support tool for faster calculation of cw train and cw test sets
     """
    if isinstance(rule_matches_z, sp.csr_matrix):
        rules_samples_ids_dict = {key: [] for key in range(rule_matches_z.shape[1])}
        for row, col in zip(*rule_matches_z.nonzero()):
            rules_samples_ids_dict[col].append(row)
    else:
        rules_samples_ids_dict = dict((i, set()) for i in range(0, rule_matches_z.shape[1]))
        for row_idx, row in enumerate(rule_matches_z):
            rules = np.where(row == 1)[0].tolist()
            for rule in rules:
                rules_samples_ids_dict[rule].add(row_idx)
    return rules_samples_ids_dict


def get_train_test_datasets_by_rule_indices(
        model_input_x: np.ndarray, rules_ids: List[int], rules_samples_ids_dict: Dict, labels: np.ndarray, fold: int,
        num_folds: int, other_sample_ids: List[int]
) -> (TensorDataset, TensorDataset):
    """
    This function returns train and test datasets for DSCrossWeigh training. Each dataloader comprises encoded
    samples, labels and sample indices in the original matrices
    :param rules_ids: shuffled rules indices
    :param labels: labels of all training samples
    :param fold: number of a current hold-out fold
    :return: dataloaders for cw training and testing
    """
    train_rules_idx, test_rules_idx = calculate_rules_indices(rules_ids, fold, num_folds)

    # select train and test samples and labels according to the selected rules idx
    test_samples, test_labels, test_idx = get_cw_samples_labels_idx(
        model_input_x, labels, test_rules_idx, rules_samples_ids_dict, check_intersections=None,
        other_sample_ids=other_sample_ids
    )

    train_samples, train_labels, _ = get_cw_samples_labels_idx(
        model_input_x, labels, train_rules_idx, rules_samples_ids_dict, check_intersections=test_idx,
        other_sample_ids=other_sample_ids
    )

    train_dataset = input_labels_to_tensordataset(train_samples, train_labels)
    test_dataset = input_info_labels_to_tensordataset(test_samples, test_idx, test_labels)

    logger.info(
        f"Fold {fold}     Rules in training set: {len(train_rules_idx)}, rules in test set: {len(test_rules_idx)}, "
        f"samples in training set: {len(train_samples)}, samples in test set: {len(test_samples)}"
    )

    return train_dataset, test_dataset


def calculate_rules_indices(rules_idx: list, fold: int, num_folds: int) -> (np.ndarray, np.ndarray):
    """
    Calculates the indices of the samples which are to be included in DSCrossWeigh training and test sets
    :param rules_idx: all rules indices (shuffled) that are to be splitted into cw training & cw test set rules
    :param fold: number of a current hold-out fold
    :return: two arrays containing indices of rules that will be used for cw training and cw test set accordingly
    """
    test_rules_idx = rules_idx[fold::num_folds]
    train_rules_idx = [rules_idx[x::num_folds] for x in range(num_folds)
                       if x != fold]
    train_rules_idx = [ids for sublist in train_rules_idx for ids in sublist]

    if not set(test_rules_idx).isdisjoint(set(train_rules_idx)):
        raise ValueError("Splitting into train and test rules is done incorrectly.")

    return train_rules_idx, test_rules_idx


def get_cw_samples_labels_idx(
        model_input_x: np.ndarray, labels: np.ndarray, indices: list, rules_samples_ids_dict: Dict,
        check_intersections: np.ndarray = None, other_sample_ids: list = None
) -> (torch.Tensor, np.ndarray, np.ndarray):
    """
    Extracts the samples and labels from the original matrices by indices. If intersection is filled with
    another sample matrix, it also checks whether the sample is not in this other matrix yet.
    :param labels: all training samples labels, shape=(num_samples, num_classes)
    :param indices: indices of rules; samples, where these rules matched & their labels are to be included in set
    :param rules_samples_ids_dict: dictionary {rule_id : sample_ids}
    :param check_intersections: optional parameter that indicates that intersections should be checked (used to
    exclude the sentences from the DSCrossWeigh training set which are already in DSCrossWeigh test set)
    :return: samples, labels and indices in the original matrix
    """
    sample_ids = [list(rules_samples_ids_dict.get(idx)) for idx in indices]
    sample_ids = list(set([value for sublist in sample_ids for value in sublist]))

    if other_sample_ids is not None:
        sample_ids = list(set(sample_ids).union(set(other_sample_ids)))

    if check_intersections is not None:
        sample_ids = return_unique(np.array(sample_ids), check_intersections)
    cw_samples_dataset = TensorDataset(torch.Tensor(model_input_x.tensors[0][sample_ids]))
    cw_labels = np.array(labels[sample_ids])
    cw_samples_idx = np.array(sample_ids)

    check_splitting(cw_samples_dataset, cw_labels, cw_samples_idx, model_input_x.tensors[0], labels)
    return cw_samples_dataset, cw_labels, cw_samples_idx