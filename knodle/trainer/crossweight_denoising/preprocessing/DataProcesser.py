import numpy as np
import pandas as pd
import utils
from commons import LABELS
import spacy
import re
from itertools import product
from typing import Union
import json

analyzer = spacy.load("en_core_web_sm")


class DataProcesser:

    def __init__(self, path_word_emb_file: str, path_input_data: str, path_patterns: str, maxlen=50):
        self.word_emb_file = path_word_emb_file
        self.path_to_input_data = path_input_data
        self.path_to_patterns = path_patterns

        self.relation2id = {"no_relation": 0}
        self.relation2patterns = {"no_relation": []}

        self.pattern2id = {}
        self.pattern2regex = {}

        self.maxlen = maxlen

    def retrieve_patterns_in_sample(self, extr_samples: list) -> Union[np.ndarray, None]:
        """
        Looks for pattern in a sample and returns a list which would be turned into a row of a Z matrix.
        :param extr_samples: list of sample substrings in the form of "ARG1 <some words> ARG2", in which the pattern
        will be searching
        :return: if there was found smth, returns a row of a Z matrix as a list, where all elements equal 0 apart from
        the element with the index corresponding to the matched pattern id - this element equals 1.
        If no pattern matched, returns None.
        """
        matched_patterns = []
        for sample, pattern_id in product(extr_samples, self.pattern2regex):
            if re.search(self.pattern2regex[pattern_id], sample):
                matched_patterns.append(pattern_id)
        if len(matched_patterns) > 0:
            return utils.get_match_matrix_row(len(self.pattern2regex), list(set(matched_patterns)))
        return None

    def get_analysed_conll_data(self) -> pd.DataFrame:
        """
        Reads conll data, extract information about sentences and gold labels. The sample are also encoded with
        glove vector idx and analysed with Spacy package in order to form the substrings for pattern search.
        :return: DataFrame with fields "samples" (raw text samples), "encoded samples" (samples encoded with glove ids),
        "retrieved patterns" (binary encoding of matched patterns in this sample), "gold_labels" (label that this
        sample got in original conll set})
        """
        word2id, id2word, word_embedding_matrix = utils.vocab_and_vectors(self.word_emb_file, ['<PAD>', '<UNK>'])
        samples, enc_samples, relations, retrieved_patterns = [], [], [], []
        with open(self.path_to_input_data, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("# id="):  # Instance starts
                    sample = ""
                    enc_tokens = []
                    label = line.split(" ")[3][5:]
                elif line == "":  # Instance ends
                    if label == "no_relation":
                        continue
                    sample_spacy = analyzer(sample).to_json()
                    sample_extractions = utils.get_extracted_sample(sample_spacy)
                    sample_patterns_retrieved = self.retrieve_patterns_in_sample(sample_extractions)

                    if sample_patterns_retrieved:
                        print(sample)
                        samples.append(sample)
                        relations.append(label)
                        enc_samples.append(np.asarray(utils.add_padding(enc_tokens, self.maxlen), dtype="float64"))
                        retrieved_patterns.append(sample_patterns_retrieved)

                elif line.startswith("#"):  # comment
                    continue
                else:
                    parts = line.split("\t")
                    token = parts[1]
                    if token == "-LRB-":
                        token = "("
                    elif token == "-RRB-":
                        token = ")"
                    sample += " " + token
                    enc_tokens.append(word2id.get(token, 1))
        return pd.DataFrame.from_dict({"samples": samples, "encoded samples": enc_samples,
                                       "retrieved patterns": retrieved_patterns, "gold_labels": relations})

    def read_pattern(self, pattern_line: str) -> list:
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
            # todo: what if the same pattern refers to different relations? now this cases are just avoided by rule
            #  first come - first serve
            return None
        if relation not in LABELS:
            return None
        relation_id = utils.get_id(relation, self.relation2id)
        pattern_id = utils.get_id(pattern, self.pattern2id)
        utils.update_dict(relation_id, pattern_id, self.relation2patterns)
        # make pattern regex
        self.pattern2regex[pattern_id] = utils.convert_pattern_to_regex(pattern)
        return utils.get_match_matrix_row(len(LABELS), [relation_id])

    def get_t_matrix(self) -> np.ndarray:
        """ Create a T matrix of pattern-relation corresponding"""
        t_matrix = []
        with open(self.path_to_patterns, encoding="UTF-8") as inp:
            for line in inp.readlines():
                pattern = self.read_pattern(line)
                if pattern:
                    t_matrix.append(pattern)
        return np.asarray(list(filter(None, t_matrix)))

    def save_dicts(self) -> None:
        """ Save all major dicts to json files"""
        with open("relations2ids.json", "w+") as f:
            json.dump(self.relation2id, f)
        with open("relation2patterns.json", "w+") as f:
            json.dump(self.relation2patterns, f)
        with open("pattern2id.json", "w+") as f:
            json.dump(self.pattern2id, f)
        with open("pattern2regex.json", "w+") as f:
            json.dump(self.pattern2regex, f)

    def process_data(self) -> None:

        t_matrix = self.get_t_matrix()
        self.save_dicts()
        samples = self.get_analysed_conll_data()

        x_matrix = np.array(list(samples["encoded samples"]), dtype=np.int)
        z_matrix = np.array(list(samples["retrieved patterns"]), dtype=np.int)

        np.save('t_matrix.npy', t_matrix)
        np.save('x_matrix.npy', x_matrix)
        np.save('z_matrix.npy', z_matrix)

        print("ok")


if __name__ == '__main__':
    DataProcesser("../data/glove.840B.300d.txt.filtered",
                  "../data/conll/dev.conll",
                  "../data/patterns.txt").process_data()

