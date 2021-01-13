import argparse
import logging
import os
import sys
from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd

from tutorials.conll_relation_extraction_dataset.commons import LABELS
from tutorials.conll_relation_extraction_dataset.utils import (get_analysed_conll_data, get_id, update_dict,
                                                               get_match_matrix_row, save_dict,
                                                               convert_pattern_to_regex)

Z_MATRIX_OUTPUT = "z_matrix.npy"
T_MATRIX_OUTPUT = "t_matrix.npy"
TRAIN_SAMPLES_OUTPUT = "train_samples.csv"
NO_PATTERN_TRAIN_SAMPLES_OUTPUT = "neg_train_samples.csv"
DEV_SAMPLES_OUTPUT = "dev_samples.csv"

logger = logging.getLogger(__name__)


class DataProcessor:

    def __init__(self,
                 path_word_emb_file: str,
                 path_train_data: str,
                 path_dev_data: str,
                 path_patterns: str,
                 path_to_output: str
                 ) -> None:
        """
        Processing the input train and dev data saved in conll format
        :param path_word_emb_file: path to file with pretrained word vectors (in our experiments: glove embeddings)
        :param path_train_data: path to data used for training and stored in conll format
        :param path_dev_data: path to data used for development and stored in conll format
        :param path_patterns: path to patterns that are going to be used for weak data supervision
        :param path_to_output: path to folder where processed data will be stored
        samples will be padded with [PAD] symbol
        """

        self.word_emb_file = path_word_emb_file
        self.path_to_train_data = path_train_data
        self.path_to_dev_data = path_dev_data
        self.path_to_patterns = path_patterns
        self.path_to_output = path_to_output

        self.relation2id = {"no_relation": 0}
        self.relation2patterns = {"no_relation": []}
        self.pattern2id = {}
        self.pattern2regex = {}

    def collect_data(self) -> None:
        """ This function reads train and dev data and saved resulted files to output directory"""
        Path(self.path_to_output).mkdir(parents=True, exist_ok=True)

        self._get_train_data()
        self._get_dev_data()

        save_dict(self.relation2id, os.path.join(self.path_to_output, "relation2id.json"))
        save_dict(self.pattern2id, os.path.join(self.path_to_output, "pattern2id.json"))

    def _get_train_data(self) -> None:
        """
        This function processes the train data and save t_matrix, z_matrix and training set info in two DataFrames:
        - DataFrame with samples where some pattern matched (samples as text, matched patterns, encoded gold labels)
        - DataFrame with samples where no pattern matched (samples as text, encoded gold labels)
        to output directory.
        """
        logger.info("Processing of train data has started")

        rule_assignments_t = self._get_t_matrix()
        train_samples, neg_train_samples = get_analysed_conll_data(self.path_to_train_data, self.pattern2regex,
                                                                   self.relation2id, logger, perform_search=True)
        rule_matches_z = np.array(list(train_samples["retrieved_patterns"]), dtype=np.int)
        self._save_train_data(rule_assignments_t, rule_matches_z, train_samples, neg_train_samples)

        logger.info("Processing of train data has finished")

    def _get_t_matrix(self) -> np.ndarray:
        """ Create a T matrix of pattern-relation correspondences"""
        t_matrix = []
        with open(self.path_to_patterns, encoding="UTF-8") as inp:
            for line in inp.readlines():
                pattern = self._read_pattern(line)
                if pattern:
                    t_matrix.append(pattern)
        return np.asarray(list(filter(None, t_matrix)))

    def _read_pattern(self, pattern_line: str) -> Union[None, list]:
        """
        Processing of pattern file line. If there is a pattern line, encode pattern and relation it corresponds to
        with ids, turn pattern into regex. Save info to corresponding dicts and return information of pattern-relation
        corresponding.
        :param pattern_line: string in pattern file
        :return: a row of future T matrix as a list
        """
        if pattern_line.startswith("#") or pattern_line == "\n":  # take only meaningful strings
            return None
        relation, pattern = pattern_line.replace("\n", "").split(" ", 1)
        if pattern in self.pattern2id:
            return None
        if relation not in LABELS:
            logger.error("Relation {} is not in TACRED relations list. The pattern will be skipped".format(relation))
            return None
        relation_id = get_id(relation, self.relation2id)
        pattern_id = get_id(pattern, self.pattern2id)
        update_dict(relation_id, pattern_id, self.relation2patterns)
        self.pattern2regex[pattern_id] = convert_pattern_to_regex(pattern)
        return get_match_matrix_row(len(LABELS), [relation_id])

    def _save_train_data(
            self, rule_assignments_t: np.ndarray, rule_matches_z: np.ndarray, train_samples: pd.DataFrame,
            neg_train_samples: pd.DataFrame
    ) -> None:
        """
        This function saves the training data to output directory
        :param rule_assignments_t: t matrix
        :param rule_matches_z: z matrix
        :param train_samples: DataFrame with samples where some pattern matched (raw samples, matched patterns,
        encoded gold labels)
        :param neg_train_samples: DataFrame with samples with no pattern matched (raw samples, encoded gold labels)
        """
        np.save(os.path.join(self.path_to_output, T_MATRIX_OUTPUT), rule_assignments_t)
        np.save(os.path.join(self.path_to_output, Z_MATRIX_OUTPUT), rule_matches_z)
        train_samples.to_csv(os.path.join(self.path_to_output, TRAIN_SAMPLES_OUTPUT), columns=["samples",
                                                                                               "raw_retrieved_patterns",
                                                                                               "labels", "enc_labels"])
        neg_train_samples.to_csv(os.path.join(self.path_to_output, NO_PATTERN_TRAIN_SAMPLES_OUTPUT),
                                 columns=["samples", "labels", "enc_labels"])

    def _get_dev_data(self) -> None:
        """
        This function processes the development data and save it as DataFrame with samples as row text and gold labels
        (encoded with ids) to output directory.
        """
        logger.info("Processing of dev data has started")

        dev_data, _ = get_analysed_conll_data(self.path_to_dev_data, self.pattern2regex, self.relation2id, logger)
        dev_data.to_csv(os.path.join(self.path_to_output, DEV_SAMPLES_OUTPUT), columns=["samples", "enc_labels",
                                                                                        "labels"])

        logger.info("Processing of dev data has finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--pretrained_embeddings", help="")
    parser.add_argument("--train_data", help="")
    parser.add_argument("--dev_data", help="")
    parser.add_argument("--patterns", help="")
    parser.add_argument("--path_to_output", help="")

    args = parser.parse_args()

    DataProcessor(
        args.pretrained_embeddings, args.train_data, args.dev_data, args.patterns, args.path_to_output).collect_data()
