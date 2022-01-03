import pandas as pd
import numpy as np

from constants import *


def z_matrix():
    """Create T-matrix from rules (mentions)."""

    mentions = pd.concat([pd.read_csv(os.path.join(MENTION_DATA_DIR, file), header=None).assign(
        classes=os.path.basename(file).split('.')[0]) for file in FILES], ignore_index=True)
    n_rules = len(mentions[0])-1

    reports = pd.read_csv(REPORTS_PATH,
                          header=None,
                          names=[REPORTS])[REPORTS].tolist()
    n_samples = len(reports)

    Z_matrix = np.zeros((n_samples, n_rules))  # , dtype=int

    return Z_matrix
