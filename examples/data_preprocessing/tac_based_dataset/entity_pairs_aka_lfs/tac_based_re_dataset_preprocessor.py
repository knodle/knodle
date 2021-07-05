import argparse
import sys
import os
from pathlib import Path
import logging
from typing import Dict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import dump

from knodle.trainer.utils import log_section
from examples.data_preprocessing.tac_based_dataset.utils.utils import count_file_lines, encode_labels

logger = logging.getLogger(__name__)
PRINT_EVERY = 1000000
Z_MATRIX_OUTPUT_TRAIN = "train_rule_matches_z.lib"
Z_MATRIX_OUTPUT_DEV = "dev_rule_matches_z.lib"
Z_MATRIX_OUTPUT_TEST = "test_rule_matches_z.lib"

T_MATRIX_OUTPUT_TRAIN = "mapping_rules_labels.lib"

TRAIN_SAMPLES_OUTPUT = "df_train.lib"
DEV_SAMPLES_OUTPUT = "df_dev.lib"
TEST_SAMPLES_OUTPUT = "df_test.lib"


def preprocess_data(
        path_train_data: str,
        path_dev_data: str,
        path_test_data: str,
        path_labels: str,
        path_lfs: str,
        path_output: str
) -> None:
    """ This function reads train and dev data and saved resulted files to output directory"""

    Path(path_output).mkdir(parents=True, exist_ok=True)

    labels2ids = get_labels(path_labels)
    num_labels = len(labels2ids)
    other_class_id = max(labels2ids.values()) + 1       # used for dev and test sets

    lfs = pd.read_csv(path_lfs)
    rule2rule_id = dict(zip(lfs.rule, lfs.rule_id))
    num_lfs = max(lfs.rule_id.values) + 1

    rule_assignments_t = get_t_matrix(lfs, num_labels)
    dump(sp.csr_matrix(rule_assignments_t), os.path.join(path_output, T_MATRIX_OUTPUT_TRAIN))

    get_train_data(
        path_train_data,
        path_output,
        rule2rule_id,
        num_lfs,
        Z_MATRIX_OUTPUT_TRAIN,
        TRAIN_SAMPLES_OUTPUT
    )

    get_dev_test_data(
        path_dev_data,
        path_output,
        labels2ids,
        rule2rule_id,
        num_lfs,
        Z_MATRIX_OUTPUT_DEV,
        DEV_SAMPLES_OUTPUT,
        other_class_id
    )

    get_dev_test_data(
        path_test_data,
        path_output,
        labels2ids,
        rule2rule_id,
        num_lfs,
        Z_MATRIX_OUTPUT_TEST,
        TEST_SAMPLES_OUTPUT,
        other_class_id
    )


def get_labels(path_labels: str) -> Dict:
    """ Reads the labels from the file and encode them with ids """
    relation2ids = {}
    with open(path_labels, encoding="UTF-8") as file:
        for line in file.readlines():
            relation, relation_enc = line.replace("\n", "").split(",")
            relation2ids[relation] = int(relation_enc)
    return relation2ids


def get_train_data(
        path_train_data: str, path_output: str, rule2rule_id: Dict, num_lfs: int, z_matrix: str, samples: str
) -> None:
    """ Processes the train data and saves t_matrix, z_matrix and training set info to output directory. """
    log_section("Processing of train data has started", logger)

    train_data = annotate_conll_data_with_lfs(path_train_data, rule2rule_id, False)
    rule_matches_z = get_z_matrix(train_data, num_lfs)

    dump(train_data, os.path.join(path_output, samples))
    dump(rule_matches_z, os.path.join(path_output, z_matrix))

    logger.info("Processing of train data has finished")


def get_dev_test_data(
        path_data: str, path_output: str, labels2ids: dict, rule2rule_id: Dict, num_lfs: int, z_matrix: str,
        samples: str, other_class_id: int
) -> None:
    """
    This function processes the development data and save it as DataFrame with samples as row text and gold labels
    (encoded with ids) to output directory. Additionally it saved z matrix for testing purposes.
    """
    log_section("Processing of eval data has started", logger)
    val_data = get_conll_data_with_ent_pairs(path_data, rule2rule_id, labels2ids, other_class_id)
    rule_matches_z = get_z_matrix(val_data, num_lfs)

    dump(rule_matches_z, os.path.join(path_output, z_matrix))
    dump(val_data, os.path.join(path_output, samples))

    logger.info("Processing of eval data has finished")


def annotate_conll_data_with_lfs(conll_data: str, rule2rule_id: Dict, filter_out_other: bool = True) -> pd.DataFrame:
    num_lines = count_file_lines(conll_data)
    processed_lines = 0
    samples, rules, enc_rules = [], [], []
    with open(conll_data, encoding='utf-8') as f:
        for line in f:
            processed_lines += 1
            line = line.strip()
            if line.startswith("# id="):  # Instance starts
                sample = ""
                subj, obj = {}, {}
            elif line == "":  # Instance ends
                if len(list(subj.keys())) == 0 or len(list(obj.keys())) == 0:
                    continue
                if min(list(subj.keys())) < min(list(obj.keys())):
                    rule = "_".join(list(subj.values())) + " " + "_".join(list(obj.values()))
                else:
                    rule = "_".join(list(subj.values())) + " " + "_".join(list(obj.values()))
                if rule in rule2rule_id.keys():
                    samples.append(sample.lstrip())
                    rules.append(rule)
                    rule_id = rule2rule_id[rule]
                    enc_rules.append(rule_id)
                elif not filter_out_other:
                    samples.append(sample.lstrip())
                    rules.append(None)
                    enc_rules.append(None)
                else:
                    continue
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
                    if (bool(subj) and not bool(obj)) or (not bool(subj) and bool(obj)):
                        sample += " " + token
            if processed_lines % PRINT_EVERY == 0:
                logger.info("Processed {:0.2f}% of {} file".format(100 * processed_lines / num_lines,
                                                                   conll_data.split("/")[-1]))

    return pd.DataFrame.from_dict({"sample": samples, "rules": rules, "enc_rules": enc_rules})


def get_conll_data_with_ent_pairs(
        conll_data: str, rule2rule_id: Dict, labels2ids: dict, other_class_id: int = None
) -> pd.DataFrame:
    """
    Processing of TACRED dataset. The function reads the .conll input file, extract the samples and the labels as well
    as argument pairs, which are saved as decision rules.
    :param conll_data: input data in .conll format
    :param rule2rule_id: corresponding of rules to rules ids
    :param labels2ids: dictionary of label - id corresponding
    :param other_class_id: id of other_class_label
    :return: DataFrame with columns "sample" (extracted sentences), "rules" (entity pairs), "enc_rules" (entity pairs
            ids), "labels" (original labels)
    """

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
                label = encode_labels(line.split(" ")[3][5:], labels2ids, other_class_id)
            elif line == "":  # Instance ends
                if len(list(subj.keys())) == 0 or len(list(obj.keys())) == 0:
                    continue
                if min(list(subj.keys())) < min(list(obj.keys())):
                    rule = "_".join(list(subj.values())) + " " + "_".join(list(obj.values()))
                else:
                    rule = "_".join(list(subj.values())) + " " + "_".join(list(obj.values()))

                if rule in rule2rule_id.keys():
                    samples.append(sample.lstrip())
                    labels.append(label)
                    rules.append(rule)
                    rule_id = rule2rule_id[rule]
                    enc_rules.append(rule_id)

                else:
                    samples.append(sample.lstrip())
                    labels.append(label)
                    rules.append(None)
                    enc_rules.append(None)

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
                    if (bool(subj) and not bool(obj)) or (not bool(subj) and bool(obj)):
                        sample += " " + token
            if processed_lines % PRINT_EVERY == 0:
                logger.info("Processed {:0.2f}% of {} file".format(
                    100 * processed_lines / num_lines, conll_data.split("/")[-1])
                )

    return pd.DataFrame.from_dict({"sample": samples, "rules": rules, "enc_rules": enc_rules, "labels": labels})


def get_t_matrix(lfs: pd.DataFrame, num_labels: int) -> np.ndarray:
    """ Function calculates t matrix (rules x labels) using the known correspondence of relations to decision rules """
    rule_assignments_t = np.empty([lfs.rule_id.max() + 1, num_labels])
    for index, row in lfs.iterrows():
        rule_assignments_t[row["rule_id"], row["label_id"]] = 1
    return rule_assignments_t


def get_z_matrix(data: pd.DataFrame, num_lfs: int) -> sp.csr_matrix:
    """
    Function calculates the z matrix (samples x rules)
    data: pd.DataFrame (samples, matched rules, matched rules id )
    output: sparse z matrix
    """
    a = data.reset_index().dropna()
    z_matrix_sparse = sp.csr_matrix((np.ones(len(a['index'].values)), (a['index'].values, a['enc_rules'].values)),
                                    shape=(len(data.index), num_lfs))
    return z_matrix_sparse


def get_max_val(_dict: dict):
    """ Returns the largest value of the dict of format {int: List[int]}"""
    return max(item for val in _dict.values() for item in val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--train_data", help="")
    parser.add_argument("--dev_data", help="")
    parser.add_argument("--test_data", help="")
    parser.add_argument("--labels", help="List of labels")
    parser.add_argument("--lfs", help="")
    parser.add_argument("--path_to_output", help="")

    args = parser.parse_args()
    preprocess_data(
        args.train_data,
        args.dev_data,
        args.test_data,
        args.labels,
        args.lfs,
        args.path_to_output
    )
