import logging
import random
from typing import List, Dict, Union

import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset

from knodle.trainer.crossweigh_weighing.utils import return_unique, check_splitting
from knodle.transformation.torch_input import input_info_labels_to_tensordataset, input_labels_to_tensordataset

logger = logging.getLogger(__name__)


def k_folds_splitting_by_rules(
        data_features: np.ndarray, labels: np.ndarray, rule_matches_z: np.ndarray, partitions: int, folds: int,
        seed: int = None, other_class_id: int = None
) -> (List, List):
    """
    This function allows to perform the splitting of the data instances into k folds according to the rules matched
    in them. The splitting could be performed in several iterations (defined by "partition" parameter).
    The logic is the following:
        for each partition:
        - the rules are shuffled
        - the rules are splitted into k folds
        - each fold iteratively becomes a hold-out fold
        - the samples that matched the rules from the hold-out fold are added to the hold-out test set
        - other samples are added to the train set
    The train and test sets do not intersect: if a sample matched several rules and one of them happened to be included
    in the test set (and the corresponding sample as well), it won't be included in the training set. Correspondingly,
    such sentences are included into several hold-out folds depending on which of the matched rules is selected to the
    hold-out fold in the current splitting.

    :param data_features: encoded data samples (num_samples x num_features)
    :param labels: array of labels (num_samples x 1)
    :param rule_matches_z: matrix of rule matches (num_samples x num_rules)
    :param partitions: number of partitions that are to be performed; in each partition the dataset will be splitted into
    k folds
    :param folds: number of folds the data instances are to be slitted into in each partition
    :param seed: optionally, the seed could be fixed in order to provide reproducibility
    :param other_class_id: if you don't want to include the negative samples (the ones that belong to the other class)
     to the test set, but only to the training set, you can pass the id of other class
    should be

    :return: two lists; the first contains the training sets, the second contains the test (hold-out) sets.
    """

    random.seed(seed) if seed is not None else random.choice(range(9999))
    rule_id2samples_ids = get_rules_sample_ids(rule_matches_z)

    return compose_train_n_test_datasets(
        data_features, rule_id2samples_ids, labels, folds, partitions, other_class_id
    )


def k_folds_splitting_by_signatures(
        data_features: np.ndarray, labels: np.ndarray, rule_matches_z: np.ndarray, partitions: int, folds: int,
        seed: int = None, other_class_id: int = None
) -> (List, List):
    """
    This function allows to perform the splitting of the data instances into k folds according to the signatures.
    The sample signature is composed from the rule matched in the sample. For example, if rules with ids 1, 5, 7 matched
    in the sample, the sample signature is 1_5_7. Thus, the signature serves as a sort of sample identification.
    The logic is the following:
        - the sample signatures are calculated
        for each partition:
        - the sample signatures are shuffled
        - the sample signatures are splitted into k folds
        - each fold iteratively becomes a hold-out fold
        - the samples with signatures from the hold-out fold are added to the hold-out test set
        - other samples are added to the train set
    The train and test sets do not intersect.

    :param data_features: encoded data samples (num_samples x num_features)
    :param labels: array of labels (num_samples x 1)
    :param rule_matches_z: matrix of rule matches (num_samples x num_rules)
    :param partitions: number of partitions that are to be performed; in each partition the dataset will be splitted into
    k folds
    :param folds: number of folds the data instances are to be slitted into in each partition
    :param seed: optionally, the seed could be fixed in order to provide reproducibility
    :param other_class_id: if you don't want to include the negative samples (the ones that belong to the other class)
     to the test set, but only to the training set, you can pass the id of other class should be

    :return: two lists; the first contains the training sets, the second contains the test (hold-out) sets.
    """

    random.seed(seed) if seed is not None else random.choice(range(9999))
    signature2id, signature_id2samples_ids = get_signature_sample_ids(rule_matches_z)

    return compose_train_n_test_datasets(
        data_features, signature_id2samples_ids, labels, folds, partitions, other_class_id
    )


def get_rules_sample_ids(rule_matches_z: Union[np.ndarray, sp.csr_matrix]) -> Dict:
    """
    This function creates a dictionary {rule id : sample id where this rule matched}. The dictionary is needed as a
    support tool for faster calculation of train and test sets.
    :param rule_matches_z: matrix of rule matches as a numpy or sparse matrix (num_samples x num_rules)
    :return: dictionary {rule_id: [sample_id1, sample_id2, ...]}
    """
    if isinstance(rule_matches_z, sp.csr_matrix):
        rules_samples_ids_dict = {key: set() for key in range(rule_matches_z.shape[1])}
        for row_id, col_id in zip(*rule_matches_z.nonzero()):       # row_id = sample id, col_id = rule id
            rules_samples_ids_dict[col_id].add(row_id)
    else:
        rules_samples_ids_dict = dict((i, set()) for i in range(0, rule_matches_z.shape[1]))
        for row_id, row in enumerate(rule_matches_z):
            rule_ids = np.where(row == 1)[0].tolist()
            for rule_id in rule_ids:
                rules_samples_ids_dict[rule_id].add(row_id)
    return rules_samples_ids_dict


def get_signature_sample_ids(rule_matches_z: np.ndarray) -> (Dict, Dict):
    """
    This function calculates the signature for each sample (e.g. if rules with ids 1, 5, 7 matched in a sample, a sample
    signature id is 1_5_7) and id for each signature.
    :param rule_matches_z: matrix of rule matches (num_samples x num_rules)
    :return: two dictionaries:
        - {signature: signature id}
        - {signature id: [sample_id1, sample_id2, ...]}
    """

    signature2id, signature_id2samples = {}, {}

    if isinstance(rule_matches_z, sp.csr_matrix):
        samples_id_rules_dict = {key: [] for key in range(rule_matches_z.shape[0])}
        for row_id, col_id in zip(*rule_matches_z.nonzero()):
            samples_id_rules_dict[row_id].append(col_id)

    else:
        samples_id_rules_dict = dict((i, set()) for i in range(0, rule_matches_z.shape[0]))
        for row_id, row in enumerate(rule_matches_z):
            samples_id_rules_dict[row_id] = np.where(row == 1)[0].tolist()

    for sample_id, rules in samples_id_rules_dict.items():
        signature = "_".join(map(str, sorted(list(rules))))

        if signature in signature2id:
            signature_id = signature2id.get(signature)
        else:
            signature_id = len(signature2id)
            signature2id[signature] = signature_id

        if signature_id in signature_id2samples:
            signature_id2samples[signature_id].append(sample_id)
        else:
            signature_id2samples[signature_id] = [sample_id]

    return signature2id, signature_id2samples


def compose_train_n_test_datasets(
        data_features: np.ndarray, rule_id2samples_ids: Dict, labels: np.ndarray, folds: int,
        partitions: int, other_class_id: int = None
) -> (List, List):
    """
    This function creates train and test datasets for k-folds cross-validation.

    :param data_features: encoded data samples (num_samples x num_features):
    :param rule_id2samples_ids: {rule_id: [sample_id1, sample_id2, ...]}. Rule in this context means anything basing on
    which splitting is performed (matched rule, sample signature, ...).
    :param labels: array of labels
    :param folds: number of folds the data instances are to be slitted into in each partition
    :param partitions: number of partitions that are to be performed
    :param other_class_id: if you don't want to include the negative samples (the ones that belong to the other class)
     to the test set, but only to the training set, you can pass the id of other class should be

    :return: list of train sets and list of corresponding test (hold-out) sets
    """

    # calculate ids of all samples that belong to the 'other_class'
    other_sample_ids = np.where(labels[:, other_class_id] == 1)[0].tolist() if other_class_id else None
    # make a list of rule ids to shuffle them later
    rule_ids = [rule_id for rule_id in range(0, len(rule_id2samples_ids))]

    train_datasets, test_datasets = [], []
    for partition in range(partitions):
        logger.info(f"Partition {partition + 1}/{partitions}:")
        random.shuffle(rule_ids)  # shuffle anew for each splitting
        for fold in range(folds):
            train_dataset, test_dataset = get_train_test_datasets_by_rule_indices(
                data_features, rule_ids, rule_id2samples_ids, labels, fold, folds, other_sample_ids
            )
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)

    return train_datasets, test_datasets


def get_train_test_datasets_by_rule_indices(
        data_features: np.ndarray, rules_ids: List[int], rule_id2samples_ids: Dict, labels: np.ndarray, fold: int,
        num_folds: int, other_sample_ids: List[int]
) -> (TensorDataset, TensorDataset):
    """
    This function returns train and test datasets for k-fold cross validation training. Each dataloader comprises
    encoded samples, labels and sample indices in the original matrices.

    :param data_features: numpy array with encoded data samples (num_samples, num_features)
    :param rules_ids: list of shuffled rules indices
    :param rule_id2samples_ids: dictionary that contains information about corresponding between rules ids and sample ids.
    Rule in this context means anything basing on which splitting is performed (matched rule, sample signature, ...)
    :param labels: labels of all training samples (num_samples, num_classes)
    :param fold: number of a current hold-out fold
    :param num_folds: the whole number of folds the data should be splitted into
    :param other_sample_ids: a list of sample ids that belong to the other class. They won't be included in the test
    set, but only to the training set.
    should be

    :return: dataloaders for cw training and testing
    """
    train_rules_idx, test_rules_idx = calculate_rules_indices(rules_ids, fold, num_folds)

    # select train and test samples and labels according to the selected rules idx
    test_samples, test_labels, test_idx = get_samples_labels_idx_by_rule_id(
        data_features, labels, test_rules_idx, rule_id2samples_ids, check_intersections=None,
        other_sample_ids=other_sample_ids
    )

    train_samples, train_labels, _ = get_samples_labels_idx_by_rule_id(
        data_features, labels, train_rules_idx, rule_id2samples_ids, check_intersections=test_idx,
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
    Calculates the indices of the samples which are to be included in training and test sets for k-fold cross validation.

    :param rules_idx: all rules indices (shuffled) that are to be splitted into cw training & cw test set rules
    :param fold: number of a current hold-out fold

    :return: two arrays containing indices of rules that will be used for cw training and cw test set accordingly
    """
    test_rules_idx = rules_idx[fold::num_folds]
    train_rules_idx = [rule_id for rule_id in rules_idx if rule_id not in test_rules_idx]

    if not set(test_rules_idx).isdisjoint(set(train_rules_idx)):
        raise ValueError("Splitting into train and test rules is done incorrectly.")

    return train_rules_idx, test_rules_idx


def get_samples_labels_idx_by_rule_id(
        data_features: np.ndarray, labels: np.ndarray, indices: list, rules_samples_ids_dict: Dict,
        check_intersections: np.ndarray = None, other_sample_ids: list = None
) -> (TensorDataset, np.ndarray, np.ndarray):
    """
    Extracts the samples and labels from the original matrices by indices. If intersection is filled with
    another sample matrix, it also checks whether the sample is not in this other matrix yet.

    :param data_features: numpy array with encoded data samples (num_samples, num_features)
    :param labels: all training samples labels (num_samples, num_classes)
    :param indices: indices of rules; samples, where these rules matched & their labels are to be included in set
    :param rules_samples_ids_dict: dictionary that contains information about corresponding from rules to sample ids.
    :param check_intersections: optional parameter that indicates that intersections should be checked (used to
    exclude the sentences from the training set which are already in the test set)
    :param other_sample_ids: a list of sample ids that belong to the other class. They won't be included in the test
    set, but only to the training set.

    :return: samples, labels and indices in the original matrix
    """
    sample_ids = [list(rules_samples_ids_dict.get(idx)) for idx in indices]
    sample_ids = list(set([value for sublist in sample_ids for value in sublist]))

    if other_sample_ids is not None:
        sample_ids = list(set(sample_ids).union(set(other_sample_ids)))

    if check_intersections is not None:
        sample_ids = return_unique(np.array(sample_ids), check_intersections)

    samples_dataset = TensorDataset(torch.Tensor(data_features.tensors[0][sample_ids]))
    samples_labels = np.array(labels[sample_ids])
    samples_idx = np.array(sample_ids)

    check_splitting(samples_dataset, samples_labels, samples_idx, data_features.tensors[0], labels)
    return samples_dataset, samples_labels, samples_idx
