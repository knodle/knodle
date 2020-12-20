from typing import Union

import numpy as np

import torch
from torch.utils.data import TensorDataset

from knodle.trainer.crossweight_denoising.preprocessing import utils
from knodle.trainer.crossweight_denoising.preprocessing.commons import LABELS

NUM_CLASSES = 39


class DataProcessor:

    def __init__(self,
                 path_word_emb_file: str,
                 path_train_data: str,
                 path_dev_data: str,
                 path_patterns: str,
                 maxlen: int = 50):

        self.word_emb_file = path_word_emb_file
        self.path_to_train_data = path_train_data
        self.path_to_dev_data = path_dev_data
        self.path_to_patterns = path_patterns

        self.relation2id = {"no_relation": 0}
        self.relation2patterns = {"no_relation": []}

        self.pattern2id = {}
        self.pattern2regex = {}

        self.maxlen = maxlen

    def collect_data(self) -> (np.ndarray, TensorDataset, np.ndarray, TensorDataset, np.ndarray, np.ndarray):

        # word2id, word_embedding_matrix = utils.vocab_and_vectors(self.word_emb_file, ['<PAD>', '<UNK>'])

        rule_assignments_t, inputs_x, rule_matches_z = self.process_train_data()
        dev_samples, dv_labels = self.process_dev_data()

        return rule_assignments_t, inputs_x, rule_matches_z, dev_samples, dv_labels

    def process_train_data(self) -> (np.ndarray, TensorDataset, np.ndarray):

        rule_assignments_t = self.get_t_matrix()

        train_samples = utils.get_analysed_conll_data(self.path_to_train_data, self.pattern2regex, perform_search=True)

        rule_matches_z = np.array(list(train_samples["retrieved_patterns"]), dtype=np.int)

        inputs_x = np.array(list(train_samples["encoded_samples"]), dtype=np.int)
        inputs_x_tensor = torch.Tensor(inputs_x)
        inputs_x_dataset = torch.utils.data.TensorDataset(inputs_x_tensor)

        return rule_assignments_t, inputs_x_dataset, rule_matches_z

    def process_dev_data(self) -> (TensorDataset, np.ndarray):

        dev_samples = utils.get_analysed_conll_data(self.path_to_dev_data)

        dev_inputs_x = np.array(list(dev_samples["encoded_samples"]), dtype=np.int)
        dev_inputs_x_tensor = torch.Tensor(dev_inputs_x)
        dev_inputs_x_dataset = torch.utils.data.TensorDataset(dev_inputs_x_tensor)

        dev_labels_y = np.array(list(dev_samples["gold_labels"]), dtype=np.int)

        return dev_inputs_x_dataset, dev_labels_y

    def get_t_matrix(self) -> np.ndarray:
        """ Create a T matrix of pattern-relation corresponding"""
        t_matrix = []
        with open(self.path_to_patterns, encoding="UTF-8") as inp:
            for line in inp.readlines():
                pattern = self.read_pattern(line)
                if pattern:
                    t_matrix.append(pattern)
        return np.asarray(list(filter(None, t_matrix)))

    def read_pattern(self, pattern_line: str) -> Union[None, list]:
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
            return None
        relation_id = utils.get_id(relation, self.relation2id)
        pattern_id = utils.get_id(pattern, self.pattern2id)
        utils.update_dict(relation_id, pattern_id, self.relation2patterns)
        self.pattern2regex[pattern_id] = utils.convert_pattern_to_regex(pattern)
        return utils.get_match_matrix_row(len(LABELS), [relation_id])


if __name__ == '__main__':
    DataProcessor("../data/glove.840B.300d.txt.filtered",
                  "../data/train.conll",
                  "../data/dev.conll",
                  "../data/patterns.txt").collect_data()
