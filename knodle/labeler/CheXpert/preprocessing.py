"""
This code builds upon the CheXpert labeler from Stanford ML Group.
It has been slightly modified to be compatible with knodle.
The original code can be found here: https://github.com/stanfordmlgroup/chexpert-labeler

----------------------------------------------------------------------------------------

Define the report preprocessing class.
"""
import bioc
import re
import pandas as pd
from negbio.pipeline import text2bioc, ssplit

from .config import CheXpertConfig


class Preprocessor:
    """
    Load and preprocess the provided report(s).

    Original code:
    https://github.com/stanfordmlgroup/chexpert-labeler/blob/master/loader/load.py
    """

    def __init__(self, config: CheXpertConfig):
        self.labeler_config = config
        self.reports_path = self.labeler_config.sample_path
        self.punctuation_spacer = str.maketrans({key: f"{key} "
                                                 for key in ".,;"})
        self.splitter = ssplit.NegBioSSplitter(newline=False)

    def preprocess(self) -> None:
        """Load and clean the report(s)."""
        collection = bioc.BioCCollection()
        reports = pd.read_csv(self.reports_path,
                              header=None,
                              names=[self.labeler_config.reports])[self.labeler_config.reports].tolist()

        for i, report in enumerate(reports):
            clean_report = self.clean(report)
            # Convert text to BioCDocument instance: id (str) = BioCDocument id, text (str): text.
            document = text2bioc.text2document(str(i), clean_report)

            split_document = self.splitter.split_doc(document)
            # If length is not exactly 1, raise error.
            assert len(split_document.passages) == 1, 'Each document must have a single passage.'

            collection.add_document(split_document)

        self.collection = collection

    def clean(self, report: pd.DataFrame = None) -> pd.DataFrame:
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
        # Clean double periods.
        clean_report = corrected_report.replace("..", ".")
        # Insert space after commas and periods.
        clean_report = clean_report.translate(self.punctuation_spacer)
        # Convert any multi white spaces to single white spaces.
        clean_report = ' '.join(clean_report.split())
        # Remove empty sentences.
        clean_report = re.sub(r'\.\s+\.', '.', clean_report)

        return clean_report
