"""Entry-point script to label radiology reports."""
import shutil

from knodle.labeler.CheXpert.stages.utils import *
from knodle.labeler.CheXpert.stages import Loader, Extractor, Classifier, Aggregator, transform


def label(transform_patterns: bool = False, uncertain: int = 1, chexpert_bool: bool = True) -> None:  # config: str = "config.py"
    """Label the provided report(s).

    Args:
        transform_patterns: Set to True if patterns are not in negbio compatible format.
        uncertain: How should uncertain matches be handled? -1 = uncertain, 0 = negative or 1 = positive
        (Info: at the moment other knodle modules can only handle 0 & 1)
        chexpert_bool: Set to True if CheXpert data is used, then some CheXpert data specific code is run.
    """
    # delimiter = "."
    # CONFIG_PATH = delimiter.join(["knodle", "labeler", "CheXpert", config])
    # from CONFIG_PATH import *

    if transform_patterns:
        transform(PRE_NEG_UNC_PATH)
        transform(NEG_PATH)
        transform(POST_NEG_UNC_PATH)

    loader = Loader()

    extractor = Extractor(chexpert_data=chexpert_bool)
    classifier = Classifier()
    aggregator = Aggregator()

    # Load reports in place.
    loader.load()
    # Extract observation mentions in place.
    extractor.extract(loader.collection)
    # Classify mentions in place.
    classifier.classify(loader.collection)

    # Adjust Z matrix.
    Z_matrix = aggregator.aggregate(loader.collection, extractor.Z_matrix, chexpert_data=chexpert_bool)

    Z_matrix[Z_matrix == UNCERTAIN] = uncertain

    # Save the matrices X, T and Z
    shutil.copy(SAMPLE_PATH, os.path.join(OUTPUT_DIR, "X_matrix.csv"))
    np.savetxt(os.path.join(OUTPUT_DIR, "T_matrix.csv"), loader.T_matrix, delimiter=",")
    np.savetxt(os.path.join(OUTPUT_DIR, "Z_matrix.csv"), Z_matrix, delimiter=",")

#
# if __name__ == "__main__":
#     parser = ArgParser()
#     label(parser.parse_args(), transform_patterns=False, uncertain=POSITIVE)
