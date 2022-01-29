"""
This code builds upon the CheXpert labeler from Stanford ML Group.
It has been slightly modified to be compatible with knodle.
The original code can be found here: https://github.com/stanfordmlgroup/chexpert-labeler

----------------------------------------------------------------------------------------

Define label updater class.
"""
import bioc
import numpy as np

from .config import CheXpertConfig
from .utils import get_rule_idx


class Updater(object):
    """
    Update Z matrix labels (positive, negative or uncertain).

    Original code: https://github.com/stanfordmlgroup/chexpert-labeler/blob/master/stages/aggregate.py
    """
    def __init__(self, config: CheXpertConfig):
        self.labeler_config = config

    def update(self,
               collection: bioc.BioCCollection,
               z_matrix: np.ndarray,
               chexpert_data: bool) -> np.ndarray:
        """For each document: Gather all labels and update Z matrix accordingly."""
        self.z_matrix = z_matrix
        NEGATION = self.labeler_config.negation
        UNCERTAINTY = self.labeler_config.uncertainty
        NEGATIVE = self.labeler_config.negative
        UNCERTAIN = self.labeler_config.uncertain
        POSITIVE = self.labeler_config.positive

        documents = collection.documents
        for i, document in enumerate(documents):

            impression_passage = document.passages[0]  # TODO: check if necessary

            for annotation in impression_passage.annotations:
                category = annotation.infons[self.labeler_config.observation]
                rule_idx = get_rule_idx(annotation.infons['term'], config=self.labeler_config)

                if NEGATION in annotation.infons:
                    label = NEGATIVE
                elif UNCERTAINTY in annotation.infons:
                    label = UNCERTAIN
                else:
                    label = POSITIVE

                # CheXpert specific checks.
                if chexpert_data:
                    # No labels are added for No Finding.
                    if category == self.labeler_config.no_finding:
                        continue

                    # Exception for 'chf' and 'heart failure' is added.
                    if ((label in [UNCERTAIN, POSITIVE]) and
                        (annotation.text == 'chf' or
                         annotation.text == 'heart failure')):
                        # The position of the first cardiomegaly rule "cardiomegaly" was manually inputted here.
                        # If there has not been a rule match -> change "0" to "-1" in the matrix & if "1", leave "1".
                        if self.z_matrix[i, 2] == NEGATIVE:
                            self.z_matrix[i, 2] = UNCERTAIN

                # Check what label has been assigned before.
                if self.z_matrix[i, rule_idx] in [label, POSITIVE]:  # if label is same as previous or previous is pos
                    continue
                elif self.z_matrix[i, rule_idx] == self.labeler_config.match:
                    self.z_matrix[i, rule_idx] = label
                elif self.z_matrix[i, rule_idx] == NEGATIVE:
                    self.z_matrix[i, rule_idx] = label
                elif self.z_matrix[i, rule_idx] == UNCERTAIN and label == POSITIVE:
                    self.z_matrix[i, rule_idx] = POSITIVE
                elif self.z_matrix[i, rule_idx] == UNCERTAIN and label == NEGATIVE:
                    self.z_matrix[i, rule_idx] = UNCERTAIN

        return self.z_matrix
