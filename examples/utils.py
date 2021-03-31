import os
from typing import Union, Tuple, List

import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer
from torch.utils.data import TensorDataset
from transformers import DistilBertTokenizer, DistilBertModel


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
    if isinstance(data, pd.Series):
        return list(data)
    elif isinstance(data, pd.DataFrame) and column_num is not None:
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


def create_bert_encoded_features(
        input_data: pd.Series, tokenizer: Tokenizer, column_num: int = None
) -> TensorDataset:
    """ Convert input data to BERT encoded features """
    encoding = tokenizer(get_samples_list(input_data, column_num), return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return TensorDataset(input_ids, attention_mask)


def create_bert_encoded_features_hidden_states(
        input_data: pd.Series, tokenizer: Tokenizer, column_num: int = None
) -> TensorDataset:
    """ Convert input data to BERT encoded features and outputs the hidden states for the input sentence """
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    encoding = tokenizer(get_samples_list(input_data, column_num), return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    output = model(input_ids, attention_mask)
    embeddings_of_last_layer = output[0]
    cls_embeddings = embeddings_of_last_layer[0]
    return cls_embeddings
