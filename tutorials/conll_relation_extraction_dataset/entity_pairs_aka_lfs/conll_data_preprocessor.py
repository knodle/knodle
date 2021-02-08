import argparse
import sys
import os
from pathlib import Path
import logging
from typing import Tuple, Dict, Union
from joblib import dump

import numpy as np
import pandas as pd

from knodle.trainer.utils import log_section
from tutorials.conll_relation_extraction_dataset.utils import (
    save_dict, count_file_lines, encode_labels, update_dict, get_id
)

logger = logging.getLogger(__name__)
PRINT_EVERY = 100000
Z_MATRIX_OUTPUT_TRAIN = "z_matrix_train.lib"
Z_MATRIX_OUTPUT_DEV = "z_matrix_dev.lib"
Z_MATRIX_OUTPUT_TEST = "z_matrix_test.lib"

T_MATRIX_OUTPUT = "t_matrix.lib"
TRAIN_SAMPLES_OUTPUT = "train_samples.csv"
DEV_SAMPLES_OUTPUT = "dev_samples.csv"

# the label for no match samples as it is in dataset; id for it will be calculated as follows: max(label_ids) + 1
OTHER_CLASS = "no_relation"
OTHER_CLASS_ID = 41


def preprocess_data(
        path_train_data: str,
        path_dev_data: str,
        path_test_data: str,
        path_labels: str,
        path_output: str
) -> None:
    """ This function reads train and dev data and saved resulted files to output directory"""

    log_section("Data processing has started", logger)
    Path(path_output).mkdir(parents=True, exist_ok=True)

    labels2ids, other_class_id = get_labels(path_labels, OTHER_CLASS)

    get_train_data(path_train_data, path_output, labels2ids, Z_MATRIX_OUTPUT_TRAIN, other_class_id)
    get_dev_test_data(path_dev_data, path_output, labels2ids, Z_MATRIX_OUTPUT_DEV)
    get_dev_test_data(path_test_data, path_output, labels2ids, Z_MATRIX_OUTPUT_TEST)

    log_section("Data processing has finished", logger)


def get_labels(path_labels: str, negative_label: str) -> Tuple[Dict, Union[int, None]]:
    """ Reads the labels from the file and encode them with ids """
    relation2ids = {}
    with open(path_labels, encoding="UTF-8") as file:
        for line in file.readlines():
            relation, relation_enc = line.replace("\n", "").split(",")
            relation2ids[relation] = int(relation_enc)
    # add no_match label
    if negative_label:
        negative_label_id = max(list(relation2ids.values())) + 1
        relation2ids[negative_label] = negative_label_id
        return relation2ids, negative_label_id
    return relation2ids, None


def get_train_data(
        path_train_data: str, path_output: str, labels2ids: dict, z_matrix: str, other_class_id: int
) -> None:
    """
    This function processes the train data and save t_matrix, z_matrix and training set info in two DataFrames:
    - DataFrame with samples where some pattern matched (samples as text, matched patterns, encoded gold labels)
    - DataFrame with samples where no pattern matched (samples as text, encoded gold labels)
    to output directory.
    """
    logger.info("Processing of train data has started")
    train_data, relation2rules, rule2id = get_conll_data_with_ent_pairs(
        path_train_data, labels2ids, True, other_class_id
    )
    rule_assignments_t = get_t_matrix(relation2rules)
    rule_matches_z = get_z_matrix(train_data)

    dump(rule_assignments_t, os.path.join(path_output, T_MATRIX_OUTPUT))
    dump(rule_matches_z, os.path.join(path_output, z_matrix))
    train_data.to_csv(os.path.join(path_output, TRAIN_SAMPLES_OUTPUT),
                      columns=["samples", "rules", "enc_rules", "labels"])
    save_dict(relation2rules, os.path.join(path_output, "relation2rules.json"))
    save_dict(rule2id, os.path.join(path_output, "rule2id.json"))

    logger.info("Processing of train data has finished")


def get_dev_test_data(path_data: str, path_output: str, labels2ids: dict, z_matrix: str) -> None:
    """
    This function processes the development data and save it as DataFrame with samples as row text and gold labels
    (encoded with ids) to output directory.
    """
    logger.info("Processing of eval data has started")
    val_data, _, _ = get_conll_data_with_ent_pairs(path_data, labels2ids)

    rule_matches_z = get_z_matrix(val_data)
    dump(rule_matches_z, os.path.join(path_output, z_matrix))
    val_data.to_csv(os.path.join(path_output, DEV_SAMPLES_OUTPUT), columns=["samples", "rules", "enc_rules", "labels"])
    logger.info("Processing of eval data has finished")


def get_conll_data_with_ent_pairs(
        conll_data: str, labels2ids: dict, filter_out_no_relation: bool = False, other_class_id: int = None
) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Processing of TACRED dataset. The function reads the .conll input file, extract the samples and the labels as well
    as argument pairs, which are saved as decision rules.
    :param conll_data: input data in .conll format
    :param labels2ids: dictionary of label - id corresponding
    :return: DataFrame with columns "samples" (extracted sentences), "rules" (entity pairs), "enc_rules" (entity pairs
            ids), "labels" (original labels)
    """

    relation2rules, rule2id = {}, {}
    num_lines = count_file_lines(conll_data)
    processed_lines = 0

    samples, labels, rules, enc_rules = [], [], [], []
    with open(conll_data, encoding='utf-8') as f:
        for line in f:
            processed_lines += 1
            line = line.strip()
            if line.startswith("# id="):  # Instance starts
                sample = ""
                subj, obj = {}, {}
                label = encode_labels(line.split(" ")[3][5:], labels2ids)
            elif line == "":  # Instance ends
                if min(list(subj.keys())) < min(list(obj.keys())):
                    rule = "_".join(list(subj.values())) + " " + "_".join(list(obj.values()))
                else:
                    rule = "_".join(list(subj.values())) + " " + "_".join(list(obj.values()))

                if filter_out_no_relation and label == other_class_id:
                    samples.append(sample)
                    labels.append(label)
                    rules.append(None)
                    enc_rules.append(None)

                elif label != "unk":
                    samples.append(sample)
                    labels.append(label)
                    rules.append(rule)

                    rule_id = get_id(rule, rule2id)
                    enc_rules.append(rule_id)
                    update_dict(label, rule_id, relation2rules)

            elif line.startswith("#"):  # comment
                continue
            else:
                splitted_line = line.split("\t")
                token = splitted_line[1]
                if splitted_line[2] == "SUBJECT":
                    subj[splitted_line[0]] = token
                    sample += " " + token
                elif splitted_line[4] == "OBJECT":
                    obj[splitted_line[0]] = token
                    sample += " " + token
                else:
                    sample += " " + token
            if processed_lines % PRINT_EVERY == 0:
                logger.info("Processed {:0.2f}% of {} file".format(100 * processed_lines / num_lines,
                                                                   conll_data.split("/")[-1]))
    return pd.DataFrame.from_dict(
        {"samples": samples,
         "rules": rules,
         "enc_rules": enc_rules,
         "labels": labels
         }), relation2rules, rule2id


def get_t_matrix(relation2rules: dict) -> np.ndarray:
    """ Function calculates t matrix (rules x labels) using the known correspondence of relations to decision rules """
    num_rules = get_max_val(relation2rules) + 1
    rule_assignments_t = np.empty([num_rules, len(relation2rules)])
    for label, rules in relation2rules.items():
        for rule in rules:
            rule_assignments_t[rule, label] = 1
    return rule_assignments_t


def get_z_matrix(train_data: pd.DataFrame) -> np.ndarray:
    """ Function calculates the z matrix (samples x rules)"""
    rules_matrix = train_data["enc_rules"].values
    rules_matrix[np.isnan(rules_matrix)] = np.nanmax(rules_matrix) + 1
    rules_matrix = (rules_matrix[:, None] == np.arange(rules_matrix.max() + 1)).astype(int)
    return rules_matrix[:, :-1]


def get_max_val(_dict: dict):
    """ Returns the largest value of the dict of format {int: List[int]}"""
    return max(item for val in _dict.values() for item in val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--train_data", help="")
    parser.add_argument("--dev_data", help="")
    parser.add_argument("--test_data", help="")
    parser.add_argument("--labels", help="List of labels")
    parser.add_argument("--path_to_output", help="")

    args = parser.parse_args()
    preprocess_data(
        args.train_data,
        args.dev_data,
        args.test_data,
        args.labels,
        args.path_to_output
    )
