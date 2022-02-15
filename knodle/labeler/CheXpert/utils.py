"""
Define helper functions.
"""
import os
import yaml

import pandas as pd
import numpy as np

from .config import CheXpertConfig


def yaml_to_df(config: CheXpertConfig) -> pd.DataFrame:

    with open(config.phrases_path) as fp:
        phrases = yaml.load(fp, yaml.FullLoader)

    phrases_list = []

    for observation, v in phrases.items():
        if 'include' in v:
            phrases_list.append([(phrases[observation]['include']), observation])

    # Sort rule row alphabetically to avoid undesired behaviour when using get_dummies()
    return pd.DataFrame(phrases_list).explode(0).sort_values(1).reset_index(drop=True)


def t_matrix_fct(config: CheXpertConfig) -> pd.DataFrame:
    """Create T-matrix from rules (mentions)."""

    phrases_df = yaml_to_df(config)

    T_matrix = pd.DataFrame(data=phrases_df).iloc[:, 1].str.get_dummies().to_numpy()

    return T_matrix


def z_matrix_fct(config: CheXpertConfig) -> np.ndarray:
    """Create Z-matrix from combination of rules (mentions) and samples."""

    phrases_df = yaml_to_df(config)

    n_rules = len(phrases_df[0])

    reports = pd.read_csv(config.sample_path,
                          header=None,
                          names=[config.reports])[config.reports].tolist()
    n_samples = len(reports)

    Z_matrix = np.zeros((n_samples, n_rules))

    return Z_matrix


def get_rule_idx(phrase: str, config: CheXpertConfig) -> pd.Int64Index:
    """Given phrase, outputs index of rule."""

    phrases_df = yaml_to_df(config)

    index = phrases_df.index[phrases_df[0] == phrase]

    return index
