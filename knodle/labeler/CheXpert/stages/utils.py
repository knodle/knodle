import os
import pandas as pd
import numpy as np
import bioc
from typing import Type
from knodle.labeler.CheXpert.config import ChexpertConfig


def t_matrix_fct(config: Type[ChexpertConfig]) -> pd.DataFrame:
    """Create T-matrix from rules (mentions)."""

    mentions = pd.concat([pd.read_csv(os.path.join(config.mention_data_dir, file), header=None).assign(
        classes=os.path.basename(file).split('.')[0]) for file in config.files], ignore_index=True)

    T_matrix = pd.DataFrame(data=mentions).iloc[:, 1].str.get_dummies().to_numpy()

    return T_matrix


def z_matrix_fct(config: Type[ChexpertConfig]) -> np.ndarray:
    """Create Z-matrix from combination of rules (mentions) and samples."""

    mentions = pd.concat([pd.read_csv(os.path.join(config.mention_data_dir, file), header=None).assign(
        classes=os.path.basename(file).split('.')[0]) for file in config.files], ignore_index=True)
    n_rules = len(mentions[0])

    reports = pd.read_csv(config.sample_path,
                          header=None,
                          names=[config.reports])[config.reports].tolist()
    n_samples = len(reports)

    Z_matrix = np.zeros((n_samples, n_rules))  # , dtype=int

    return Z_matrix


def get_rule_idx(phrase: str, config: Type[ChexpertConfig]) -> int:
    """Given phrase, outputs index of rule."""

    mentions = pd.concat([pd.read_csv(os.path.join(config.mention_data_dir, file), header=None).assign(
        classes=os.path.basename(file).split('.')[0]) for file in config.files], ignore_index=True)

    index = mentions.index[mentions[0] == phrase]

    return index


def transform(text_file: str) -> None:
    """Transform file of words to patterns which are compatible with ngrex."""

    file = open(text_file, "r+")

    new_file = []

    for line in file:
        lemmatized1 = "{} < {} {lemma:/" + str(line).rstrip() + "/}"
        new_file.append(lemmatized1)

        lemmatized2 = "{} > {} {lemma:/" + str(line).rstrip() + "/}"
        new_file.append(lemmatized2)

    with open(text_file, "w+") as file:
        for i in new_file:
            file.write(i + "\n")
