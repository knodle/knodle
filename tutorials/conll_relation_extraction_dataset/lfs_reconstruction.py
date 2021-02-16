import argparse
import csv
import os
import sys
from pathlib import Path
import logging
from typing import Dict

from tutorials.conll_relation_extraction_dataset.utils import (
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

    with open(os.path.join(path_output, "lfs.csv"), 'a', newline='') as csvfile:
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


def get_lfs(conll_data: str, labels2ids: Dict) -> list:
    relation2rules, rule2id = {}, {}
    rules = []
    num_lines = count_file_lines(conll_data)
    processed_lines = 0

    with open(conll_data, encoding='utf-8') as f:
        sample_num = 0
        for line in f:
            processed_lines += 1
            line = line.strip()
            if line.startswith("# id="):  # Instance starts
                sample_num += 1
                subj, obj = {}, {}
                label = convert_to_tacred_rel(line.split(" ")[3][5:])
                if label not in labels2ids and label != "no_relation":
                    logger.info(label)
                label_id = encode_labels(label, labels2ids)
            elif line == "":  # Instance ends
                if len(list(subj.keys())) == 0 or len(list(obj.keys())) == 0:
                    continue
                if min(list(subj.keys())) < min(list(obj.keys())):
                    rule = "_".join(list(subj.values())) + " " + "_".join(list(obj.values()))
                else:
                    rule = "_".join(list(subj.values())) + " " + "_".join(list(obj.values()))

                if label_id != "unk":
                    rule_id = get_id(rule, rule2id)
                    update_dict(label, rule_id, relation2rules)
                    rule_info = [str(rule), str(rule_id), str(label), str(label_id)]
                    if rule_info not in rules:
                        rules.append(rule_info)

            elif line.startswith("#"):  # comment
                continue
            else:
                splitted_line = line.split("\t")
                token = splitted_line[1]
                if splitted_line[2] == "SUBJECT":
                    subj[splitted_line[0]] = token
                elif splitted_line[4] == "OBJECT":
                    obj[splitted_line[0]] = token
            if processed_lines % PRINT_EVERY == 0:
                logger.info("Processed {:0.2f}% of {} file".format(100 * processed_lines / num_lines,
                                                                   conll_data.split("/")[-1]))
    return rules


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
