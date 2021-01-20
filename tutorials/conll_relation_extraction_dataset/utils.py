import pandas as pd
import numpy as np
import re
import itertools
from typing import Union
import torch
import spacy

ARG1 = "$ARG1"
ARG2 = "$ARG2"

disable_cuda = True
device = None
if not disable_cuda and torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")


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
        (ARG1 + sample["text"][ent1["end"] : ent2["start"]] + ARG2)
        if ent1["end"] < ent2["end"]
        else (ARG2 + sample["text"][ent2["end"] : ent1["start"]] + ARG1)
        for ent1, ent2 in itertools.permutations(sample["ents"], 2)
    ]


def get_analysed_conll_data(
    conll_data: str, patterns2regex: Union[dict, None], perform_search: bool = False
) -> pd.DataFrame:
    """
    Reads conll data, extract information about sentences and gold labels. The sample are analysed with SpaCy package.
    :param conll_data: path to data saved in conll format
    :param patterns2regex: dictionary with pattern and their corresponding regexes
    :param perform_search: boolean whether indicates whether also pattern search in sentences is to be performed
    :return: DataFrame with fields "samples" (raw text samples), "gold_labels" (label that samples got in original
    conll set}) and "retrieved patterns" (matched patterns in samples; empty if pattern search wasn't performed)
    """
    samples, enc_samples, relations, retrieved_patterns = [], [], [], []
    analyzer = spacy.load("en_core_web_sm")
    with open(conll_data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# id="):  # Instance starts
                sample = ""
                label = line.split(" ")[3][5:]
            elif line == "":  # Instance ends
                if label == "no_relation":
                    continue
                sample_spacy = analyzer(sample).to_json()
                sample_extractions = get_extracted_sample(sample_spacy)

                if perform_search:
                    sample_patterns_retrieved = retrieve_patterns_in_sample(
                        sample_extractions, patterns2regex
                    )
                    if sample_patterns_retrieved:
                        samples.append(sample)
                        relations.append(label)
                        retrieved_patterns.append(sample_patterns_retrieved)
                else:
                    samples.append(sample)
                    relations.append(label)
                    retrieved_patterns.append([])

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
    return pd.DataFrame.from_dict(
        {
            "samples": samples,
            "retrieved_patterns": retrieved_patterns,
            "gold_labels": relations,
        }
    )


def retrieve_patterns_in_sample(
    extr_samples: list, pattern2regex: dict
) -> Union[np.ndarray, None]:
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
        return get_match_matrix_row(len(pattern2regex), list(set(matched_patterns)))
    return None
