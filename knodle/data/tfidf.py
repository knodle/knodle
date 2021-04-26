from typing import List, Union, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np

from knodle.data.transformer import get_samples_list


def create_tfidf_values(text_data: [str], max_features: int = 2000) -> np.array:
    """Takes text and transforms it.

    :param text_data: List of text strings
    :param max_features: Maximum number of features for the Vectorizer
    :return: Tensordataset, holding Tf-Idf values
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    transformed_data = vectorizer.fit_transform(text_data).toarray().astype(np.float)

    return vectorizer, TensorDataset(torch.from_numpy(transformed_data))


def create_tfidf_input(
        train_x: List[str],
        rule_matches_z: np.array,
        mapping_rules_labels_t: np.array,
        train_y: np.array,
        test_x: List[str],
        test_y: np.array,
        num_features: int = 2000,
) -> [TensorDataset, np.array, np.array, TensorDataset, TensorDataset, TensorDataset]:
    """Transform data for trianing purposes.

    :param train_x: List of training texts
    :param rule_matches_z: Matrix as described in paper
    :param mapping_rules_labels_t: T matrix
    :param train_y: Gold labels
    :param test_x: List of testing texts
    :param test_y: Gold labels
    :param num_features: Number of features
    :return: Data needed for an arbitrary training run based on tf-idf
    """

    vectorizer = TfidfVectorizer(max_features=num_features)
    train_x = vectorizer.fit_transform(train_x).toarray().astype(np.float)
    train_x = TensorDataset(torch.from_numpy(train_x))

    train_rule_matches_z = rule_matches_z.astype(np.float)
    if train_y is not None:
        train_y = torch.from_numpy(train_y)

    test_x = vectorizer.transform(test_x).toarray().astype(np.float)
    test_x = TensorDataset(torch.from_numpy(test_x))
    test_y = torch.from_numpy(test_y)

    return (
        train_x,
        train_rule_matches_z,
        mapping_rules_labels_t,
        train_y,
        test_x,
        test_y,
    )


def get_tfidf_features(
        train_data: List, test_data: List = None, dev_data: List = None
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, None]]:
    """
    Convert input data to a matrix of TF-IDF features.
    :param train_data: training samples that are to be encoded with TF-IDF features. Can be given as Series or
    as DataFrames with specified column number where the sample are stored.
    :param column_num: optional parameter that is needed to specify in which column of input_data Dataframe the samples
    are stored
    :param test_data: if DataFrame/Series with test data is provided
    :param dev_data: if DataFrame/Series with development data is provided, it will be encoded as well
    :return: TensorDataset with encoded data
    """
    dev_transformed_data, test_transformed_data = None, None
    vectorizer = TfidfVectorizer()

    train_transformed_data = vectorizer.fit_transform(train_data)
    if test_data is not None:
        test_transformed_data = vectorizer.transform(test_data)
    if dev_data is not None:
        dev_transformed_data = vectorizer.transform(dev_data)

    return train_transformed_data, test_transformed_data, dev_transformed_data
