import logging
import os
import sys

import pandas as pd
from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer


def read_train_dev_test(target_path: str):
    train_df = load(os.path.join(target_path, 'train_df.lib'))
    dev_df = load(os.path.join(target_path, 'dev_df.lib'))
    test_df = load(os.path.join(target_path, 'test_df.lib'))
    train_rule_matches_z = load(os.path.join(target_path, 'train_rule_matches_z.lib')).toarray()
    dev_rule_matches_z = load(os.path.join(target_path, 'dev_rule_matches_z.lib')).toarray()
    test_rule_matches_z = load(os.path.join(target_path, 'test_rule_matches_z.lib')).toarray()
    mapping_rules_labels_t = load(os.path.join(target_path, 'mapping_rules_labels_t.lib'))
    imdb_dataset = pd.read_csv(os.path.join(target_path, 'imdb_data_preprocessed.csv'))

    return train_df, dev_df, test_df, train_rule_matches_z, dev_rule_matches_z, test_rule_matches_z, imdb_dataset, \
           mapping_rules_labels_t


def create_tfidf_values(
        text_data: [str], force_create_new: bool = False, max_features: int = None
):
    if os.path.exists("tutorials/ImdbDataset/tfidf.lib") and not force_create_new:
        cached_data = load("tutorials/ImdbDataset/tfidf.lib")
        if cached_data.shape == text_data.shape:
            return cached_data

    vectorizer = TfidfVectorizer(min_df=2, max_features=max_features)
    transformed_data = vectorizer.fit_transform(text_data)
    dump(transformed_data, "tutorials/ImdbDataset/tfidf.lib")
    return transformed_data


def init_logger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
