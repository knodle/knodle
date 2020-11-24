import numpy as np
from snorkel.labeling.model import MajorityLabelVoter


def get_majority_vote_probabilities(
    rule_matches: np.ndarray, output_classes: int
) -> np.ndarray:
    majority_model = MajorityLabelVoter(cardinality=output_classes)
    prediction_probabilities = majority_model.predict_proba(rule_matches)
    return prediction_probabilities
