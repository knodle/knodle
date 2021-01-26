import pandas as pd
import numpy as np
from sklearn.metrics.classification import classification_report

from knodle.trainer.utils.denoise import get_majority_vote_labels


def get_z_stats(z_matrix: np.array):
    pass


def majority_vote(z, t, y):
    y_majority = get_majority_vote_labels(z, t)
    return classification_report(y, y_majority, output_dict=True)



