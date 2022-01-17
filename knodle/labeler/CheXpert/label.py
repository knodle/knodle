"""Entry-point script to label radiology reports."""
import shutil

from knodle.labeler.CheXpert.config import ChexpertConfig
from knodle.labeler.CheXpert.stages.utils import *
from knodle.labeler.CheXpert.stages import Loader, Extractor, Classifier, Aggregator, transform


class Labeler:

    def __init__(self, **kwargs):
        self.labeler_config = kwargs.get("labeler_config", ChexpertConfig())

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

        loader = Loader(config=self.labeler_config)

        extractor = Extractor(config=self.labeler_config, chexpert_data=chexpert_bool)
        classifier = Classifier(config=self.labeler_config)
        aggregator = Aggregator(config=self.labeler_config)

        # Load reports in place.
        loader.load()
        # Extract observation mentions in place.
        extractor.extract(loader.collection)
        # Classify mentions in place.
        classifier.classify(loader.collection)

        # Adjust Z matrix.
        Z_matrix = aggregator.aggregate(loader.collection, extractor.Z_matrix, chexpert_data=chexpert_bool)

        Z_matrix[Z_matrix == self.labeler_config.uncertain] = uncertain

        # Save the matrices X, T and Z
        shutil.copy(self.labeler_config.sample_path, os.path.join(self.labeler_config.output_dir, "X_matrix.csv"))
        np.savetxt(os.path.join(self.labeler_config.output_dir, "T_matrix.csv"), loader.T_matrix, delimiter=",")
        np.savetxt(os.path.join(self.labeler_config.output_dir, "Z_matrix.csv"), Z_matrix, delimiter=",")
