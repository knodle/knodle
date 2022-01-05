"""Define report loader class."""
import re
import bioc
from negbio.pipeline import text2bioc, ssplit
from . import t_matrix_fct

from constants import *


class Loader:
    """Report loader."""
    def __init__(self):
        self.reports_path = REPORTS_PATH
        # add space after punctuation symbols: I added ";" here
        self.punctuation_spacer = str.maketrans({key: f"{key} "
                                                 for key in ".,;"})
        self.splitter = ssplit.NegBioSSplitter(newline=False)

    def load(self):
        """Load and clean the reports."""
        collection = bioc.BioCCollection()
        reports = pd.read_csv(self.reports_path,
                              header=None,
                              names=[REPORTS])[REPORTS].tolist()
        # self.X_matrix = []

        for i, report in enumerate(reports):
            clean_report = self.clean(report)
            # self.X_matrix.append(clean_report)
            # convert text to BioCDocument instance: id (str) = BioCDocument id, text (str): text
            document = text2bioc.text2document(str(i), clean_report)

            split_document = self.splitter.split_doc(document)

            # if length is not exactly 1, raise error
            assert len(split_document.passages) == 1, 'Each document must have a single passage.'

            collection.add_document(split_document)

        self.collection = collection
        self.T_matrix = t_matrix_fct()

    def clean(self, report):
        """Clean the report text."""
        lower_report = report.lower()
        # Change `and/or` to `or`.
        corrected_report = re.sub('and/or',
                                  'or',
                                  lower_report)
        # Change any `XXX/YYY` to `XXX or YYY`.
        corrected_report = re.sub('(?<=[a-zA-Z])/(?=[a-zA-Z])',
                                  ' or ',
                                  corrected_report)
        # Clean double periods
        clean_report = corrected_report.replace("..", ".")
        # Insert space after commas and periods.
        clean_report = clean_report.translate(self.punctuation_spacer)
        # Convert any multi white spaces to single white spaces.
        clean_report = ' '.join(clean_report.split())
        # Remove empty sentences
        clean_report = re.sub(r'\.\s+\.', '../CheXpert', clean_report)

        return clean_report
