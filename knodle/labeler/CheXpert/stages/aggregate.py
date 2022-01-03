"""Define mention aggregator class."""
import numpy as np
from tqdm import tqdm

from constants import *


class Aggregator(object):
    """Aggregate mentions of observations from radiology reports."""
    def __init__(self, rules, verbose=False):
        self.rules = rules

        self.verbose = verbose

    def dict_to_vec(self, d):
        """
        Convert a dictionary of the form

        {cardiomegaly: [1],
         opacity: [u, 1],
         fracture: [0]}

        into a vector of the form

        [np.nan, np.nan, 1, u, np.nan, ..., 0, np.nan]
        """
        vec = []
        for rule in self.rules:
            # There was a mention of the rule.
            if rule in d:
                label_list = d[rule]
                # Only one label, no conflicts.
                if len(label_list) == 1:
                    vec.append(label_list[0])
                # Multiple labels.
                else:
                    # Case 1. There is negated and uncertain.
                    if NEGATIVE in label_list and UNCERTAIN in label_list:
                        vec.append(UNCERTAIN)
                    # Case 2. There is negated and positive.
                    elif NEGATIVE in label_list and POSITIVE in label_list:
                        vec.append(POSITIVE)
                    # Case 3. There is uncertain and positive.
                    elif UNCERTAIN in label_list and POSITIVE in label_list:
                        vec.append(POSITIVE)
                    # Case 4. All labels are the same.
                    else:
                        vec.append(label_list[0])

            # No mention of the rule
            else:
                vec.append(np.nan)

        return vec

    def aggregate(self, collection, z_matrix):
        self.Z_matrix = z_matrix

        labels = []
        documents = collection.documents
        if self.verbose:
            print("Aggregating mentions...")
            documents = tqdm(documents)
        for i, document in enumerate(documents):
            label_dict = {}
            impression_passage = document.passages[0]
            no_finding = True
            for annotation in impression_passage.annotations:
                category = annotation.infons[OBSERVATION]
                rule = annotation.infons['term']

                if NEGATION in annotation.infons:
                    label = NEGATIVE
                elif UNCERTAINTY in annotation.infons:
                    label = UNCERTAIN
                else:
                    label = POSITIVE

                # If at least one non-support category has a uncertain or
                # positive label, there was a finding
                if (category != SUPPORT_DEVICES and
                    label in [UNCERTAIN, POSITIVE]):
                    no_finding = False

                # Don't add any labels for No Finding
                if category == NO_FINDING:
                    continue

                # add exception for 'chf' and 'heart failure'
                if ((label in [UNCERTAIN, POSITIVE]) and
                    (annotation.text == 'chf' or
                     annotation.text == 'heart failure')):
                    if CARDIOMEGALY not in label_dict:
                        label_dict[CARDIOMEGALY] = [UNCERTAIN]
                    else:
                        label_dict[CARDIOMEGALY].append(UNCERTAIN)

                if rule not in label_dict:
                    label_dict[rule] = [label]
                else:
                    label_dict[rule].append(label)

            # need to get rid of NO FINDING
            # if no_finding:
            #     label_dict[NO_FINDING] = [POSITIVE]

            label_vec = self.dict_to_vec(label_dict)

            labels.append(label_vec)

            self.Z_matrix[i, :] = label_vec

        return self.Z_matrix
