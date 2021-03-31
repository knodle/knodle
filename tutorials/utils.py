import os
from typing import Union, Tuple, List

import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer


def read_train_dev_test(
        target_path: str, if_dev_data: bool = False
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
           Tuple[pd.DataFrame, None, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]]:
    """This function loads the matrices as well as train, test (occasionally, also dev) data from corresponding files"""
    df_train = load(os.path.join(target_path, 'df_train.lib'))
    df_test = load(os.path.join(target_path, 'df_test.lib'))
    z_train_rule_matches = load(os.path.join(target_path, 'z_train_rule_matches.lib'))
    z_test_rule_matches = load(os.path.join(target_path, 'z_test_rule_matches.lib'))
    t_mapping_rules_labels = load(os.path.join(target_path, 't_mapping_rules_labels.lib'))

    if if_dev_data:
        dev_df = load(os.path.join(target_path, 'df_dev.lib'))
        return df_train, dev_df, df_test, z_train_rule_matches, z_test_rule_matches, t_mapping_rules_labels

    return df_train, None, df_test, z_train_rule_matches, z_test_rule_matches, t_mapping_rules_labels


def get_samples_list(data: Union[pd.Series, pd.DataFrame], column_num: int = None) -> List:
    """ Extracts the data from the Series/DataFrame and returns it as a list"""
    if isinstance(data, pd.Series):
        return list(data)
    elif isinstance(data, pd.DataFrame) and column_num:
        return list(data.iloc[:, column_num])
    else:
        raise ValueError(
            "Please pass input data either as a Series or as a DataFrame with number of the column with samples"
        )


def create_tfidf_values(text_data: [str], max_features, path_to_cash: str = None):
    if path_to_cash and os.path.exists(path_to_cash):
        cached_data = load(path_to_cash)
        if cached_data.shape == text_data.shape:
            return cached_data

    vectorizer = TfidfVectorizer(max_features=max_features)
    transformed_data = vectorizer.fit_transform(text_data)
    dump(transformed_data, path_to_cash)
    return transformed_data
