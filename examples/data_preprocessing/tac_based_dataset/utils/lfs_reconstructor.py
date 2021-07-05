import argparse
import csv
import os
import sys
from pathlib import Path
import logging
from typing import Dict, Tuple, List

from tutorials.data_preprocessing.tac_based_dataset.conll_relation_extraction_dataset.utils import (
    count_file_lines, get_id, update_dict, convert_to_tacred_rel
)

logger = logging.getLogger(__name__)
PRINT_EVERY = 1000000


def reconstruct_arg_pairs_lfs(
        path_data: str,
        path_labels: str,
        path_output: str
) -> None:
    """ Thins function reconstructs argument pairs that have been LFs while constructing the conll noisy dataset """
    Path(path_output).mkdir(parents=True, exist_ok=True)
    labels2ids = get_labels(path_labels)
    rules = get_lfs(path_data, labels2ids)

    with open(os.path.join(path_output, "lfs_no_overlapping_rules.csv"), 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["rule", "rule_id", "label", "label_id"])
        writer.writeheader()
        writer = csv.writer(csvfile)
        writer.writerows(rules)


def get_labels(path_labels: str) -> Dict:
    """ Reads the labels from the file and encode them with ids """
    relation2ids = {}
    with open(path_labels, encoding="UTF-8") as file:
        for line in file.readlines():
            relation, relation_enc = line.replace("\n", "").split(",")
            relation2ids[relation] = int(relation_enc)
    return relation2ids


def get_rule(
        subj: list, obj: list, subj_min_token_id: int, obj_min_token_id: int, label, label_id, rule2id, relation2rules, rules
):
    if subj_min_token_id < obj_min_token_id:
        rule = "_".join(subj) + " " + "_".join(obj)
    else:
        rule = "_".join(obj) + " " + "_".join(subj)
    if label_id != "unk":
        rule_id = get_id(rule, rule2id)
        update_dict(label, rule_id, relation2rules)
        rule_info = [str(rule), str(rule_id), str(label), str(label_id)]
        if rule_info not in rules:
            rules.append(rule_info)
    return rules


def extract_subj_obj(
        line: str, subj: list, obj: list, subj_min_token_id: int, obj_min_token_id: int
) -> Tuple[List, List, int, int]:
    splitted_line = line.split("\t")
    token = splitted_line[1]
    if splitted_line[2] == "SUBJECT":
        if not subj_min_token_id:
            subj_min_token_id = int(splitted_line[0])
        subj.append(token)
    elif splitted_line[4] == "OBJECT":
        if not obj_min_token_id:
            obj_min_token_id = int(splitted_line[0])
        obj.append(token)
    return subj, obj, subj_min_token_id, obj_min_token_id


def get_lfs(conll_data: str, labels2ids: Dict) -> list:
    relation2rules, rule2id = {}, {}
    rules = []
    num_lines = count_file_lines(conll_data)
    processed_lines = 0
    with open(conll_data, encoding='utf-8') as f:
        for line in f:
            processed_lines += 1
            line = line.strip()
            if line.startswith("# id="):  # Instance starts
                subj, obj = [], []
                subj_min_token_id, obj_min_token_id = 0, 0
                label, label_id = get_label_n_label_id(line, labels2ids)
            elif line == "":  # Instance ends
                if len(subj) == 0 or len(obj) == 0:
                    continue
                rules = get_rule(
                    subj, obj, subj_min_token_id, obj_min_token_id, label, label_id, rule2id, relation2rules, rules
                )
            elif line.startswith("#"):  # comment
                continue
            else:
                subj, obj, subj_min_token_id, obj_min_token_id = extract_subj_obj(
                    line, subj, obj, subj_min_token_id, obj_min_token_id
                )
            print_progress(processed_lines, num_lines)
    return rules


def get_label_n_label_id(line: str, labels2ids: dict) -> Tuple[str, int]:
    label = convert_to_tacred_rel(line.split(" ")[3][5:])
    if label not in labels2ids and label != "no_relation":
        logger.info(label)
    label_id = encode_labels(label, labels2ids)
    return label, label_id


def print_progress(processed_lines: int, num_lines: int) -> None:
    if processed_lines % (int(round(num_lines / 10))) == 0:
        print(f"Processed {processed_lines / num_lines * 100 :0.0f}%")


def encode_labels(label: str, label2id: dict) -> int:
    """ Encodes labels with corresponding labels id. If relation is unknown, returns special id for unknown relations"""
    return label2id.get(label, "unk")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--path_data", help="")
    parser.add_argument("--path_labels", help="")
    parser.add_argument("--path_outputs", help="")

    args = parser.parse_args()
    reconstruct_arg_pairs_lfs(
        args.path_data,
        args.path_labels,
        args.path_outputs
    )
