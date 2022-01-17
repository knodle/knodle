"""Define mention aggregator class."""
from .utils import *


class Aggregator(object):
    """Aggregate mentions of observations from reports."""
    def __init__(self, config: Type[ChexpertConfig]):
        self.labeler_config = config

    def aggregate(self,
                  collection: Type[bioc.BioCCollection],
                  z_matrix: np.ndarray,
                  chexpert_data: bool) -> np.ndarray:
        self.Z_matrix = z_matrix
        NEGATION = self.labeler_config.negation
        NEGATIVE = self.labeler_config.negative
        UNCERTAINTY = self.labeler_config.uncertainty
        UNCERTAIN = self.labeler_config.uncertain
        POSITIVE = self.labeler_config.positive

        documents = collection.documents
        for i, document in enumerate(documents):

            impression_passage = document.passages[0]

            for annotation in impression_passage.annotations:
                category = annotation.infons[self.labeler_config.observation]
                rule_idx = get_rule_idx(annotation.infons['term'], config=self.labeler_config)

                if NEGATION in annotation.infons:
                    label = NEGATIVE
                elif UNCERTAINTY in annotation.infons:
                    label = UNCERTAIN
                else:
                    label = POSITIVE

                # CheXpert specific checks
                if chexpert_data:
                    # No labels are added for No Finding.
                    if category == self.labeler_config.no_finding:
                        continue

                    # Exception for 'chf' and 'heart failure' is added.
                    if ((label in [UNCERTAIN, POSITIVE]) and
                        (annotation.text == 'chf' or
                         annotation.text == 'heart failure')):
                        # The position of the first cardiomegaly rule "cardiomegaly" was manually inputted here.
                        # If there has not been a rule match -> change 0 to -1 in the matrix & if 1, leave 1.
                        if self.Z_matrix[i, 2] == NEGATIVE:
                            self.Z_matrix[i, 2] = UNCERTAIN

                # It is checked what label has been assigned before.
                if self.Z_matrix[i, rule_idx] in [label, POSITIVE]:  # if label is same as previous or previous is POS
                    continue
                elif self.Z_matrix[i, rule_idx] == self.labeler_config.match:
                    self.Z_matrix[i, rule_idx] = label
                elif self.Z_matrix[i, rule_idx] == NEGATIVE:
                    self.Z_matrix[i, rule_idx] = label
                elif self.Z_matrix[i, rule_idx] == UNCERTAIN and label == POSITIVE:
                    self.Z_matrix[i, rule_idx] = POSITIVE
                elif self.Z_matrix[i, rule_idx] == UNCERTAIN and label == NEGATIVE:
                    self.Z_matrix[i, rule_idx] = UNCERTAIN

        return self.Z_matrix
