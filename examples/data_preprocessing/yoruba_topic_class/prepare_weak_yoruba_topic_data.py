"""
Weak Supervision Dataset for Topic Classification in Yorùbá

This is based on the paper

Hedderich, Adelani, Zhu, Alabi, Markus & Klakow:
Transfer Learning and Distant Supervision for Multilingual Transformer Models: A Study on African Languages
EMNLP 2020, https://aclanthology.org/2020.emnlp-main.204/

It is a topic classification task of news articles for a low-resource language. The weak supervision uses
keyword matching rules for the topics. The original authors used their own keyword conflict resolution strategy and
only provided one label per instance. This is a re-implementation of their keyword matching that provides all
matching rules for each instance.

The labels are "nigeria", "africa", "world", "politics", "sport", "entertainment", "health"
See the paper and the code below for more details.

@author: Michael A. Hedderich
@version: 1.0
"""

import json
import re
import os
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import nltk
from nltk.util import ngrams
import joblib

DATA_TRAIN_URL = "https://raw.githubusercontent.com/uds-lsv/transfer-distant-transformer-african/master/data/yoruba_newsclass/train_clean.tsv"
DATA_DEV_URL = "https://raw.githubusercontent.com/uds-lsv/transfer-distant-transformer-african/master/data/yoruba_newsclass/dev.tsv"
DATA_TEST_URL = "https://raw.githubusercontent.com/uds-lsv/transfer-distant-transformer-african/master/data/yoruba_newsclass/test.tsv"
KEYWORD_LISTS_DIR_URL = "https://raw.githubusercontent.com/uds-lsv/transfer-distant-transformer-african/master/data/yoruba_newsclass/lexicon/"
DATA_CSV_SEPARATOR = "\t"
DATA_PATH = "./"

LABEL_NAMES = ["nigeria", "africa", "world", "politics", "sport", "entertainment", "health"]


def load_remote_files(label_names):
    train = pd.read_csv(DATA_TRAIN_URL, sep=DATA_CSV_SEPARATOR)
    dev = pd.read_csv(DATA_DEV_URL, sep=DATA_CSV_SEPARATOR)
    test = pd.read_csv(DATA_TEST_URL, sep=DATA_CSV_SEPARATOR)
    keyword_lists = load_remote_keyword_lists(label_names)
    return train, dev, test, keyword_lists


def load_remote_keyword_lists(label_names):
    keyword_lists = {}
    for label_name in label_names:
        # the keyword lists are just textfiles, one keyword per line
        keyword_list = urllib.request.urlopen(KEYWORD_LISTS_DIR_URL + label_name + ".txt").readlines()
        keyword_list = [token.decode().strip() for token in keyword_list]
        keyword_lists[label_name] = set(keyword_list)

    return keyword_lists


def extract_ngrams(text, length):
    """ Converting a text string into n-grams of the given length.
        Implementation taken from the original authors.
    """
    n_grams = ngrams(nltk.word_tokenize(text), length)
    return [' '.join(grams) for grams in n_grams]


def create_keyword_match_rule(keyword_list):
    """ Return a function that returns true if there is an overlap between
        the keyword_list and a given text
    """
    def matches(text):
        return len(text.intersection(keyword_list)) > 0
    return matches


def create_rule_matches_matrix(data, rules):
    """ Connection between instance (data) and label (rules)
        Creates the z-matrix: #instances x #rules
    """
    matrix = np.zeros((len(data), len(rules)))

    for i, row in data.iterrows():
        text = row["news_title"]
        text = text.lower()
        text = re.sub("[,.;?!:]", " ", text)
        text = re.sub(" +", " ", text)

        text_onegrams = extract_ngrams(text, 1)
        text_twograms = extract_ngrams(text, 2)
        text = set(text_onegrams + text_twograms)

        for j, rule in enumerate(rules):
            if rule(text):  # if rule matches this instance
                matrix[i][j] = 1

    return matrix


def show_examples(data, rule_match_matrix):
    print(f"For sentence '{data.iloc[1]['news_title']}' the following rules match {rule_match_matrix[1]}")


def save(train, dev, test, train_rule_matrix, dev_rule_matrix, test_rule_matrix, rule_label_matrix, data_path):
    # Knodle expects the input text to be called "sample"
    train.rename(columns={"news_title": "sample"}, inplace=True)
    dev.rename(columns={"news_title": "sample"}, inplace=True)
    test.rename(columns={"news_title": "sample"}, inplace=True)

    train.to_csv(os.path.join(data_path, "df_train.csv"), index=None)
    dev.to_csv(os.path.join(data_path, "df_dev.csv"), index=None)
    test.to_csv(os.path.join(data_path, "df_test.csv"), index=None)

    joblib.dump(train, os.path.join(data_path, "df_train.lib"))
    joblib.dump(dev, os.path.join(data_path, "df_dev.lib"))
    joblib.dump(test, os.path.join(data_path, "df_test.lib"))

    joblib.dump(train_rule_matrix, os.path.join(data_path, "train_rule_matches_z.lib"))
    joblib.dump(dev_rule_matrix, os.path.join(data_path, "dev_rule_matches_z.lib"))
    joblib.dump(test_rule_matrix, os.path.join(data_path, "test_rule_matches_z.lib"))

    joblib.dump(rule_label_matrix, os.path.join(data_path, "mapping_rules_labels_t.lib"))

    info = {"creation time": datetime.now().strftime("%Y/%m/%d, %H:%M"),
            "creation tool": "prepare_weak_yoruba_topic_data.py - Knodle example script - version 1.0"}
    json.dump(info, open(os.path.join(data_path, "data_info.json"), "w"))


def main():
    label_names = LABEL_NAMES

    # load the data from the authors' GitHub
    df_train, df_dev, df_test, keyword_lists = load_remote_files(label_names)
    rules = [create_keyword_match_rule(keyword_lists[label_name]) for label_name in label_names]

    # connection between instance and label (z-matrix; #instances x #rules)
    # matrix that shows for each instance (row) which keyword rule (column) matches
    # 1 for a match, 0 otherwise
    train_rule_matches_matrix = create_rule_matches_matrix(df_train, rules)
    dev_rule_matches_matrix = create_rule_matches_matrix(df_dev, rules)
    test_rule_matches_matrix = create_rule_matches_matrix(df_dev, rules)

    # connection between rule and label (t-matrix; #rules x #labels)
    # each rule is specific for one label/class, so we can just use the identity matrix
    rule_label_matrix = np.ones((len(rules), len(rules)))

    # some outputs as examples
    print(f"Label order {', '.join(str(i) + ':' + label for i, label in enumerate(label_names))}")
    show_examples(df_train, train_rule_matches_matrix)
    show_examples(df_dev, dev_rule_matches_matrix)
    show_examples(df_test, test_rule_matches_matrix)

    # store everything
    save(df_train, df_dev, df_test, train_rule_matches_matrix, dev_rule_matches_matrix,
         test_rule_matches_matrix, rule_label_matrix, DATA_PATH)


if __name__ == "__main__":
    main()
