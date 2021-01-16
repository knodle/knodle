import argparse
import logging
import os
import re
import sys
from typing import Union
from pathlib import Path
from joblib import dump

import numpy as np
import pandas as pd

from tutorials.conll_relation_extraction_dataset.utils import (get_analysed_conll_data, get_id, update_dict,
                                                               get_match_matrix_row, save_dict, convert_pattern_to_regex)

Z_MATRIX_OUTPUT = "z_matrix.npy"
T_MATRIX_OUTPUT = "t_matrix.npy"
TRAIN_SAMPLES_OUTPUT = "train_samples.csv"
NO_PATTERN_TRAIN_SAMPLES_OUTPUT = "neg_train_samples.csv"
DEV_SAMPLES_OUTPUT = "dev_samples.csv"

logger = logging.getLogger(__name__)
relation2id, pattern2id, pattern2regex, relation2patterns = {}, {}, {}, {}


def collect_data(path_train_data: str, path_dev_data: str, path_patterns: str, path_labels: str, path_output: str) -> None:
    """ This function reads train and dev data and saved resulted files to output directory"""
    Path(path_output).mkdir(parents=True, exist_ok=True)

    labels = _get_labels(path_labels)

    _get_train_data(path_train_data, path_patterns, path_output, labels)
    _get_dev_data(path_dev_data, path_output)

    save_dict(relation2id, os.path.join(path_output, "relation2id.json"))
    save_dict(pattern2id, os.path.join(path_output, "pattern2id.json"))


def _get_labels(path_labels) -> list:
    """ This function reads the labels from the file"""
    with open(path_labels, encoding="UTF-8") as file:
        return [re.sub("\n", "", line) for line in file.readlines()]


def _get_train_data(path_train_data: str, path_patterns: str, path_output: str, labels: list) -> None:
    """
    This function processes the train data and save t_matrix, z_matrix and training set info in two DataFrames:
    - DataFrame with samples where some pattern matched (samples as text, matched patterns, encoded gold labels)
    - DataFrame with samples where no pattern matched (samples as text, encoded gold labels)
    to output directory.
    """
    logger.info("Processing of train data has started")

    rule_assignments_t = _get_t_matrix(path_patterns, labels)
    train_samples, neg_train_samples = get_analysed_conll_data(path_train_data, pattern2regex, relation2id, logger,
                                                               perform_search=True)
    rule_matches_z = np.array(list(train_samples["retrieved_patterns"]), dtype=np.int)
    _save_train_data(rule_assignments_t, rule_matches_z, train_samples, neg_train_samples, path_output)

    logger.info("Processing of train data has finished")


def _get_t_matrix(path_patterns: str, labels: list) -> np.ndarray:
    """ Create a T matrix of pattern-relation correspondences"""
    t_matrix = []
    with open(path_patterns, encoding="UTF-8") as inp:
        for line in inp.readlines():
            pattern = _read_pattern(line, labels)
            if pattern:
                t_matrix.append(pattern)
    return np.asarray(list(filter(None, t_matrix)))


def _read_pattern(pattern_line: str, labels: list) -> Union[None, list]:
    """
    Processing of pattern file line. If there is a pattern line, encode pattern and relation it corresponds to
    with ids, turn pattern into regex. Save info to corresponding dicts and return information of pattern-relation
    corresponding.
    :return: a row of future T matrix as a list
    """
    if pattern_line.startswith("#") or pattern_line == "\n":  # take only meaningful strings
        return None
    relation, pattern = pattern_line.replace("\n", "").split(" ", 1)
    if pattern in pattern2id:
        return None
    if relation not in labels:
        logger.error("Relation {} is not in TACRED relations list. The pattern will be skipped".format(relation))
        return None
    relation_id = get_id(relation, relation2id)
    pattern_id = get_id(pattern, pattern2id)
    update_dict(relation_id, pattern_id, relation2patterns)
    pattern2regex[pattern_id] = convert_pattern_to_regex(pattern)
    return get_match_matrix_row(len(labels), [relation_id])


def _save_train_data(
        rule_assignments_t: np.ndarray, rule_matches_z: np.ndarray, train_samples: pd.DataFrame,
        neg_train_samples: pd.DataFrame, path_output: str
) -> None:
    """
    This function saves the training data to output directory
    :param rule_assignments_t: t matrix
    :param rule_matches_z: z matrix
    :param train_samples: DataFrame with samples where some pattern matched (raw samples, matched patterns,
    encoded gold labels)
    :param neg_train_samples: DataFrame with samples with no pattern matched (raw samples, encoded gold labels)
    """
    dump(rule_assignments_t, os.path.join(path_output, T_MATRIX_OUTPUT))
    dump(rule_matches_z, os.path.join(path_output, Z_MATRIX_OUTPUT))
    train_samples.to_csv(os.path.join(path_output, TRAIN_SAMPLES_OUTPUT), columns=["samples", "raw_retrieved_patterns",
                                                                                   "labels", "enc_labels"])
    neg_train_samples.to_csv(os.path.join(path_output, NO_PATTERN_TRAIN_SAMPLES_OUTPUT),
                             columns=["samples", "labels", "enc_labels"])


def _get_dev_data(path_dev_data: str, path_output: str) -> None:
    """
    This function processes the development data and save it as DataFrame with samples as row text and gold labels
    (encoded with ids) to output directory.
    """
    logger.info("Processing of dev data has started")

    dev_data, _ = get_analysed_conll_data(path_dev_data, pattern2regex, relation2id, logger)
    dev_data.to_csv(os.path.join(path_output, DEV_SAMPLES_OUTPUT), columns=["samples", "enc_labels", "labels"])

    logger.info("Processing of dev data has finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--train_data", help="")
    parser.add_argument("--dev_data", help="")
    parser.add_argument("--patterns", help="")
    parser.add_argument("--labels", help="List of labels")
    parser.add_argument("--path_to_output", help="")

    args = parser.parse_args()
    collect_data(args.train_data, args.dev_data, args.patterns, args.labels, args.path_to_output)
