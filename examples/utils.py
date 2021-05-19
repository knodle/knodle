import os
from numbers import Integral
from typing import Union, List, Tuple

import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import TensorDataset


def read_train_dev_test(
        target_path: str, if_dev_data: bool = False
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
           Tuple[pd.DataFrame, None, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]]:
    """This function loads the matrices as well as train, test (occasionally, also dev) data from corresponding files"""
    df_train = load(os.path.join(target_path, 'df_train.lib'))
    df_test = load(os.path.join(target_path, 'df_test.lib'))
    z_train_rule_matches = load(os.path.join(target_path, 'train_rule_matches_z.lib'))
    z_test_rule_matches = load(os.path.join(target_path, 'test_rule_matches_z.lib'))
    t_mapping_rules_labels = load(os.path.join(target_path, 'mapping_rules_labels_t.lib'))

    if if_dev_data:
        dev_df = load(os.path.join(target_path, 'df_dev.lib'))
        return df_train, dev_df, df_test, z_train_rule_matches, z_test_rule_matches, t_mapping_rules_labels

    return df_train, None, df_test, z_train_rule_matches, z_test_rule_matches, t_mapping_rules_labels


def get_samples_list(data: Union[pd.Series, pd.DataFrame], column_num: int = None) -> List:
    """ Extracts the data from the Series/DataFrame and returns it as a list"""
    column_num = int(column_num)
    if isinstance(data, pd.Series):
        return list(data)
    elif isinstance(data, pd.DataFrame) and column_num is not None:
        return list(data.iloc[:, column_num])
    else:
        raise ValueError(
            "Please pass input data either as a Series or as a DataFrame with number of the column with samples"
        )


def get_tfidf_features(
        train_data: List, test_data: List = None, dev_data: List = None, path_to_cache: str = None,
        max_features: int = None
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, None]]:
    """
    Convert input data to a matrix of TF-IDF features.
    :param train_data: training samples that are to be encoded with TF-IDF features. Can be given as Series or
    as DataFrames with specified column number where the sample are stored.
    :param column_num: optional parameter that is needed to specify in which column of input_data Dataframe the samples
    are stored
    :param test_data: if DataFrame/Series with test data is provided
    :param dev_data: if DataFrame/Series with development data is provided, it will be encoded as well
    :param path_to_cache: a path to the folder where calculated cached TF-IDF values should be saved
    :param max_features: If not None, build a vocabulary that only consider the top max_features ordered by term
    frequency across the corpus.
    :return: TensorDataset with encoded data
    """

    dev_transformed_data, test_transformed_data = None, None
    vectorizer = TfidfVectorizer(max_features=max_features)

    train_transformed_data = vectorizer.fit_transform(train_data)
    if test_data is not None:
        test_transformed_data = vectorizer.transform(test_data)
    if dev_data is not None:
        dev_transformed_data = vectorizer.transform(dev_data)

    if path_to_cache:
        dump(train_transformed_data, path_to_cache)
        dump(dev_transformed_data, path_to_cache)
        dump(test_transformed_data, path_to_cache)

    return train_transformed_data, test_transformed_data, dev_transformed_data


def convert_text_to_transformer_input(tokenizer, texts: List[str]) -> TensorDataset:
    """
    Convert input data to BERT encoded features (more details could be found at
    https://huggingface.co/transformers/model_doc)
    :param texts: training/dev/test samples that are to be encoded with BERT features. Can be given as Series or
    as DataFrames with specified column number where the sample are stored.
    :param tokenizer: DistilBertTokenizer tokenizer for english from HuggingFace
    :param column_num: optional parameter that is needed to specify in which column of input_data Dataframe the samples are stored
    :return: TensorDataset with encoded data
    """
    encoding = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding.get('input_ids')
    attention_mask = encoding.get('attention_mask')

    input_values_x = TensorDataset(input_ids, attention_mask)

    return input_values_x
