import pandas as pd
import os

from constants import *


def t_matrix():
    """Create T-matrix from rules (mentions)."""

    mentions = pd.concat([pd.read_csv(os.path.join(MENTION_DATA_DIR, file), header=None).assign(
        classes=os.path.basename(file).split('.')[0]) for file in FILES], ignore_index=True)

    T_matrix = pd.DataFrame(data=mentions).iloc[:, 1].str.get_dummies().to_numpy()

    return T_matrix
