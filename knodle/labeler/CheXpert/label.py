"""
This code builds upon the CheXpert labeler from Stanford ML Group.
It has been slightly modified to be compatible with knodle.
The original code can be found here: https://github.com/stanfordmlgroup/chexpert-labeler

----------------------------------------------------------------------------------------
TEST
Entry-point script to label the provided reports.
"""
import os
import shutil
import numpy as np

from .config import CheXpertConfig
from .utils import t_matrix_fct
from . import Preprocessor, Matcher, NegUncDetector, Updater
from knodle.labeler.labeler import Labeler


class CheXpertLabeler(Labeler):

    def __init__(self, **kwargs):
        if kwargs.get("labeler_config", None) is None:
            kwargs["labeler_config"] = CheXpertConfig()
        super().__init__(**kwargs)

    def label(self, uncertain: int = 1, chexpert_bool: bool = True) -> None:
        """
        Label the provided report(s).

        Args:
            uncertain: How should uncertain matches be handled? "-1" = uncertain, "0" = negative or "1" = positive
            [Info: at the moment other knodle modules can only handle "0" (negative) & "1" (positive)]
            chexpert_bool: Set to "True" if CheXpert data is used, then some CheXpert data specific code is run.
        """

        preprocessor = Preprocessor(config=self.labeler_config)
        matcher = Matcher(config=self.labeler_config, chexpert_data=chexpert_bool)
        neg_unc_detector = NegUncDetector(config=self.labeler_config)
        updater = Updater(config=self.labeler_config)

        # Get reports & preprocess them.
        preprocessor.preprocess()
        # Get mention & unmention phrases, look for matches.
        matcher.match(preprocessor.collection)
        # Get patterns and detect negative/uncertain matches.
        neg_unc_detector.neg_unc_detect(preprocessor.collection)

        # Update Z matrix.
        z_matrix = updater.update(preprocessor.collection, matcher.z_matrix, chexpert_data=chexpert_bool)

        # Change uncertain labels to positive or negative, if specified in function call.
        if uncertain != self.labeler_config.uncertain:
            z_matrix[z_matrix == self.labeler_config.uncertain] = uncertain

        # Create T matrix.
        t_matrix = t_matrix_fct(config=self.labeler_config)

        # Save the matrices X, T and Z.
        shutil.copy(self.labeler_config.sample_path, os.path.join(self.labeler_config.output_dir, "X_matrix.csv"))
        np.savetxt(os.path.join(self.labeler_config.output_dir, "T_matrix.csv"), t_matrix, delimiter=",")
        np.savetxt(os.path.join(self.labeler_config.output_dir, "Z_matrix.csv"), z_matrix, delimiter=",")
