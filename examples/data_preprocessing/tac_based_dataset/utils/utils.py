import itertools
import json
import logging
import re
from typing import Union, Tuple

import spacy
import pandas as pd
from pandas import DataFrame

ARG1 = "$ARG1"
ARG2 = "$ARG2"
PRINT_EVERY = 10000
UNKNOWN_RELATIONS_ID = 404  # id which will be assigned to unknown relations, i.e. relations that wasn't seen in KB

logger = logging.getLogger(__name__)


def get_analysed_conll_data(
        conll_data: str,
        patterns2regex: Union[dict, None],
        labels2ids: dict,
        perform_search: bool = False,
) -> Tuple[DataFrame, DataFrame]:
    """
    Reads conll data, extract information about sentences and gold labels. The sample are analysed with SpaCy package
    :param labels2ids: dictionary with labels and their ids
    :param conll_data: path to data saved in conll format
    :param patterns2regex: dictionary with pattern and their corresponding regexes
    :param perform_search: boolean whether indicates whether also pattern search in sentences is to be performed
    :return: DataFrame with fields "sample" (raw text samples), "gold_labels" (label that samples got in original
    conll set}) and "retrieved patterns" (matched patterns in samples; empty if pattern search wasn't performed)
    """
    all_lines = count_file_lines(conll_data)
    processed_lines = 0
    analyzer = spacy.load("en_core_web_sm")

    (
        samples,
        labels,
        enc_labels,
        patterns_retr,
        raw_patterns_retr,
        neg_samples,
        neg_labels,
        neg_enc_labels,
    ) = ([] for _ in range(8))

    with open(conll_data, encoding="utf-8") as f:
        for line in f:
            processed_lines += 1
            line = line.strip()
            if line.startswith("# id="):  # Instance starts
                sample = ""
                label = line.split(" ")[3][5:]
                enc_label = encode_labels(label, labels2ids)
            elif line == "":  # Instance ends
                if (
                        label == "no_relation" or enc_label == UNKNOWN_RELATIONS_ID
                ):  # skip no_relation samples or unknown rel
                    continue
                sample_spacy = analyzer(sample).to_json()
                sample_extractions = get_extracted_sample(sample_spacy)

                if perform_search:
                    (
                        raw_pattern_retrieved,
                        encoded_pattern_retrieved,
                    ) = retrieve_patterns_in_sample(sample_extractions, patterns2regex)
                    if encoded_pattern_retrieved:
                        samples.append(sample)
                        labels.append(label)
                        enc_labels.append(enc_label)
                        patterns_retr.append(encoded_pattern_retrieved)
                        raw_patterns_retr.append(raw_pattern_retrieved)
                    else:  # if nothing is found, add this sample preserving its original label
                        neg_samples.append(sample)
                        neg_labels.append(label)
                        neg_enc_labels.append(enc_label)
                else:
                    samples.append(sample)
                    labels.append(label)
                    enc_labels.append(enc_label)
                    patterns_retr.append([])
                    raw_patterns_retr.append([])

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
            if processed_lines % PRINT_EVERY == 0:
                logger.info(
                    "Processed {:0.2f}% of {} file".format(
                        100 * processed_lines / all_lines, conll_data.split("/")[-1]
                    )
                )
    return build_df(
        samples,
        patterns_retr,
        raw_patterns_retr,
        labels,
        enc_labels,
        neg_samples,
        neg_labels,
        neg_enc_labels,
    )


def encode_labels(label: str, label2id: dict, other_class: int) -> int:
    """ Encodes labels with corresponding labels id. If relation is unknown, returns special id for unknown relations"""
    return label2id.get(label, other_class)


def build_df(
        samples: list,
        retrieved_patterns: list,
        raw_patterns_retrieved: list,
        labels: list,
        enc_labels: list,
        neg_samples: list,
        neg_labels: list,
        neg_enc_labels: list,
) -> (pd.DataFrame, pd.DataFrame):
    """
    This function builds two dataframes: one with samples where some rule matched (columns: sample, matches pattern,
    original label) and the second with labels that got no rule matches (columns: sample, label=original label)
    """
    samples = pd.DataFrame.from_dict(
        {
            "sample": samples,
            "retrieved_patterns": retrieved_patterns,
            "raw_retrieved_patterns": raw_patterns_retrieved,
            "labels": labels,
            "enc_labels": enc_labels,
        }
    )
    no_pattern_samples = pd.DataFrame.from_dict(
        {"sample": neg_samples, "enc_labels": neg_enc_labels, "labels": neg_labels}
    )
    return samples, no_pattern_samples


def get_id(item: Union[int, str], dic: dict) -> Union[int, str]:
    """This function checks if there is a key in a dict. If so, it returns a value of a dict.
    Creates a dictionary pair {key : max(dict value) + 1} otherwise"""
    if item in dic:
        item_id = dic[item]
    else:
        item_id = len(dic)
        dic[item] = item_id
    return item_id


def update_dict(key: Union[int, str], value: Union[int, str], dic: dict) -> None:
    """This function checks if there is a key in a dict. If so, it added new value to the list of values for this key.
    Create a {key: {value}} pair otherwise."""
    if key in dic:
        dic[key].append(value)
    else:
        dic[key] = [value]


def convert_pattern_to_regex(pattern: str) -> str:
    """ Transform the pattern into regex by char escaping and adding of optional articles in front of the arguments"""
    return re.sub("\\\\\\$ARG", "(A )?(a )?(The )?(the )?\\$ARG", re.escape(pattern))


def get_match_matrix_row(num_columns: int, matched_columns: list) -> list:
    """
    There is a function that creates a list which will be transformed into matrix row in future. All elements in the
    list equal zeros, apart from the ones with indices given in matched_columns list - they equal ones.
    :param num_columns: length of created list = future row vector 2nd dimensionality
    :param matched_columns: list of element indices that equal to 1
    :return: binary list
    """
    matrix_row = [0] * num_columns
    if matched_columns:
        for m_column in matched_columns:
            matrix_row[m_column] = 1
    return matrix_row


def get_extracted_sample(sample: dict) -> list:
    """
    Using the Spacy entity recognition information, forms the subtractions of the samples in form of
    "ARG1 <some words> ARG2" strings which will be used to look for patterns in them
    :param sample: sample analysed with SpaCy package and saved as dictionary
    :return: list of sample substrings
    """
    return [
        (ARG1 + sample["text"][ent1["end"]: ent2["start"]] + ARG2)
        if ent1["end"] < ent2["end"]
        else (ARG2 + sample["text"][ent2["end"]: ent1["start"]] + ARG1)
        for ent1, ent2 in itertools.permutations(sample["ents"], 2)
    ]


def retrieve_patterns_in_sample(
        extr_samples: list, pattern2regex: dict
) -> Union[Tuple[None, None], Tuple[list, list]]:
    """
    Looks for pattern in a sample and returns a list which would be turned into a row of a Z matrix.
    :param extr_samples: list of sample substrings in the form of "ARG1 <some words> ARG2", in which the patterns
    will be searching
    :param pattern2regex: dictionary with pattern and their corresponding regexes
    :return: if there was found smth, returns a row of a Z matrix as a list, where all elements equal 0 apart from
    the element with the index corresponding to the matched pattern id - this element equals 1.
    If no pattern matched, returns None.
    """
    matched_patterns = []
    for sample, pattern in itertools.product(extr_samples, pattern2regex):
        if re.search(pattern2regex[pattern], sample):
            matched_patterns.append(pattern)
    if len(matched_patterns) > 0:
        return matched_patterns, get_match_matrix_row(
            len(pattern2regex), list(set(matched_patterns))
        )
    return None, None


def count_file_lines(file_name: str) -> int:
    """ Count the number of line in a file """
    with open(file_name) as f:
        return len(f.readlines())


def save_dict(dict_: dict, path: str) -> None:
    with open(path, "w+") as f:
        json.dump(dict_, f)


def convert_to_tacred_rel(label: str) -> str:
    """ Function changes some labels in train data that are transcribed differently to tacred labels"""
    if label == "org:top_members_employees":
        return "org:top_members/employees"
    elif label == "org:political_religious_affiliation":
        return "org:political/religious_affiliation"
    elif label == "per:employee_or_member_of":
        return "per:employee_of"
    elif label == "org:number_of_employees_members":
        return "org:number_of_employees/members"
    elif label == "per:statesorprovinces_of_residence":
        return "per:stateorprovinces_of_residence"
    elif label == "org:date_founded":
        return "org:founded"
    elif label == "org:date_dissolved":
        return "org:dissolved"
    else:
        return label
