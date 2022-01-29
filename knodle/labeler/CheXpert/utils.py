"""
Define helper functions.
"""
import os
import pandas as pd
import numpy as np

from .config import CheXpertConfig


def t_matrix_fct(config: CheXpertConfig) -> pd.DataFrame:
    """Create T-matrix from rules (mentions)."""

    mentions = pd.concat([pd.read_csv(os.path.join(config.mention_data_dir, file), header=None).assign(
        classes=os.path.basename(file).split('.')[0]) for file in config.files], ignore_index=True)

    T_matrix = pd.DataFrame(data=mentions).iloc[:, 1].str.get_dummies().to_numpy()

    return T_matrix


def z_matrix_fct(config: CheXpertConfig) -> np.ndarray:
    """Create Z-matrix from combination of rules (mentions) and samples."""

    mentions = pd.concat([pd.read_csv(os.path.join(config.mention_data_dir, file), header=None).assign(
        classes=os.path.basename(file).split('.')[0]) for file in config.files], ignore_index=True)
    n_rules = len(mentions[0])

    reports = pd.read_csv(config.sample_path,
                          header=None,
                          names=[config.reports])[config.reports].tolist()
    n_samples = len(reports)

    Z_matrix = np.zeros((n_samples, n_rules))

    return Z_matrix


def get_rule_idx(phrase: str, config: CheXpertConfig) -> int:
    """Given phrase, outputs index of rule."""

    mentions = pd.concat([pd.read_csv(os.path.join(config.mention_data_dir, file), header=None).assign(
        classes=os.path.basename(file).split('.')[0]) for file in config.files], ignore_index=True)

    index = mentions.index[mentions[0] == phrase]

    return index
