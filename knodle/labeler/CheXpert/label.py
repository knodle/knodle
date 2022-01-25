"""Entry-point script to label radiology reports."""
import shutil

from knodle.labeler.CheXpert.config import ChexpertConfig
from knodle.labeler.CheXpert.stages.utils import *
from knodle.labeler.CheXpert.stages import Preprocessor, Matcher, Finetuner, Updater, transform


class Labeler:

    # this does not work
    def __init__(self, **kwargs):
        if kwargs.get("labeler_config", None) is None:
            kwargs["labeler_config"] = ChexpertConfig()
        super().__init__(**kwargs)
        # this works:
        # if kwargs.get("labeler_config", None) is None:
        #     self.labeler_config = kwargs.get("labeler_config", ChexpertConfig())

    def label(self, transform_patterns: bool = False, uncertain: int = 1, chexpert_bool: bool = True) -> None:
        """Label the provided report(s).

        Args:
            transform_patterns: Set to True if patterns are not in negbio compatible format.
            uncertain: How should uncertain matches be handled? -1 = uncertain, 0 = negative or 1 = positive
            (Info: at the moment other knodle modules can only handle 0 (negative) & 1 (positive))
            chexpert_bool: Set to True if CheXpert data is used, then some CheXpert data specific code is run.
        """

        if transform_patterns:
            transform(self.labeler_config.pre_neg_unc_path)
            transform(self.labeler_config.neg_path)
            transform(self.labeler_config.post_neg_unc_path)

        preprocessor = Preprocessor(config=self.labeler_config)

        matcher = Matcher(config=self.labeler_config, chexpert_data=chexpert_bool)
        finetuner = Finetuner(config=self.labeler_config)
        updater = Updater(config=self.labeler_config)

        # Get reports & preprocess them.
        preprocessor.preprocess()
        # Get mention & unmention phrases, look for matches.
        matcher.match(preprocessor.collection)
        # Get patterns and finetune matches as negative/uncertain.
        finetuner.finetune(preprocessor.collection)

        # Update Z matrix.
        Z_matrix = updater.update(preprocessor.collection, matcher.Z_matrix, chexpert_data=chexpert_bool)

        Z_matrix[Z_matrix == self.labeler_config.uncertain] = uncertain

        # Save the matrices X, T and Z
        shutil.copy(self.labeler_config.sample_path, os.path.join(self.labeler_config.output_dir, "X_matrix.csv"))
        np.savetxt(os.path.join(self.labeler_config.output_dir, "T_matrix.csv"), preprocessor.T_matrix, delimiter=",")
        np.savetxt(os.path.join(self.labeler_config.output_dir, "Z_matrix.csv"), Z_matrix, delimiter=",")
