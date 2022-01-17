"""Define mention aggregator class."""
from .utils import *


class Aggregator(object):
    """Aggregate mentions of observations from radiology reports."""
    def aggregate(self, collection, z_matrix: np.ndarray, chexpert_data: bool) -> np.ndarray:  # todo: check types
        self.Z_matrix = z_matrix

        documents = collection.documents
        for i, document in enumerate(documents):

            impression_passage = document.passages[0]

            for annotation in impression_passage.annotations:
                category = annotation.infons[OBSERVATION]
                rule_idx = get_rule_idx(annotation.infons['term'])

                if NEGATION in annotation.infons:
                    label = NEGATIVE
                elif UNCERTAINTY in annotation.infons:
                    label = UNCERTAIN
                else:
                    label = POSITIVE

                # CheXpert specific checks
                if chexpert_data:
                    # Don't add any labels for No Finding.
                    if category == NO_FINDING:
                        continue

                    # Add exception for 'chf' and 'heart failure'.
                    if ((label in [UNCERTAIN, POSITIVE]) and
                        (annotation.text == 'chf' or
                         annotation.text == 'heart failure')):
                        # Manually inputted the positions of the first cardiomegaly rule "cardiomegaly"
                        # if there is no rule match -> change 0 to -1 in the matrix; if 1, leave 1
                        if self.Z_matrix[i, 2] == NEGATIVE:
                            self.Z_matrix[i, 2] = UNCERTAIN

                # Check what label has been assigned before.
                if self.Z_matrix[i, rule_idx] in [label, POSITIVE]:  # if label is same as previous or previous is POS
                    continue
                elif self.Z_matrix[i, rule_idx] == MATCH:
                    self.Z_matrix[i, rule_idx] = label
                elif self.Z_matrix[i, rule_idx] == NEGATIVE:
                    self.Z_matrix[i, rule_idx] = label
                elif self.Z_matrix[i, rule_idx] == UNCERTAIN and label == POSITIVE:
                    self.Z_matrix[i, rule_idx] = POSITIVE
                elif self.Z_matrix[i, rule_idx] == UNCERTAIN and label == NEGATIVE:
                    self.Z_matrix[i, rule_idx] = UNCERTAIN

        return self.Z_matrix
