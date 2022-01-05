"""Define mention aggregator class."""
from tqdm import tqdm
from . import get_rule_idx

from constants import *


class Aggregator(object):
    """Aggregate mentions of observations from radiology reports."""
    def __init__(self, verbose=False):
        self.verbose = verbose

    def aggregate(self, collection, z_matrix):
        self.Z_matrix = z_matrix

        documents = collection.documents
        if self.verbose:
            print("Aggregating mentions...")
            documents = tqdm(documents)
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

                # Don't add any labels for No Finding
                if category == NO_FINDING:
                    continue

                # add exception for 'chf' and 'heart failure'
                if ((label in [UNCERTAIN, POSITIVE]) and
                    (annotation.text == 'chf' or
                     annotation.text == 'heart failure')):
                    # manually inputted the positions of the first cardiomegaly rule "cardiomegaly"
                    # if there is no rule match -> change 0 to -1 in the matrix; if 1, leave 1
                    if self.Z_matrix[i, 2] == NEGATIVE:
                        self.Z_matrix[i, 2] = UNCERTAIN

                # check what label has been assigned before
                if self.Z_matrix[i, rule_idx] in [label, POSITIVE]:  # if label is same as previous or previous is 1
                    continue
                elif self.Z_matrix[i, rule_idx] == 999:
                    self.Z_matrix[i, rule_idx] = label
                elif self.Z_matrix[i, rule_idx] == NEGATIVE:
                    self.Z_matrix[i, rule_idx] = label
                elif self.Z_matrix[i, rule_idx] == UNCERTAIN and label == POSITIVE:
                    self.Z_matrix[i, rule_idx] = POSITIVE
                elif self.Z_matrix[i, rule_idx] == UNCERTAIN and label == NEGATIVE:
                    self.Z_matrix[i, rule_idx] = UNCERTAIN

        return self.Z_matrix
