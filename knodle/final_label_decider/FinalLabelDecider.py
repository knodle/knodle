
import numpy as np
from snorkel.labeling.model import MajorityLabelVoter


def get_majority_vote_probabilities(applied_lfs: np.ndarray, output_classes: int) -> np.ndarray:
    majority_model = MajorityLabelVoter(cardinality=output_classes)
    prediction_probabilities = majority_model.predict_proba(applied_lfs)
    return prediction_probabilities

def get_label_model_probabilities(applied_lfs: np.ndarray, **kwargs) -> np.ndarray:
    pass