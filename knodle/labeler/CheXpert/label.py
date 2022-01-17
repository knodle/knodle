"""Entry-point script to label radiology reports."""
import shutil

from knodle.labeler.CheXpert.stages.utils import *
from knodle.labeler.CheXpert.stages import Loader, Extractor, Classifier, Aggregator, transform


# Should uncertain matches be POSITIVE=1, NEGATIVE=0 or UNCERTAIN=-1?
# Info: at the moment other knodle modules can only handle 0 & 1
def label(transform_patterns: bool = False, uncertain: int = 1, chexpert_bool: bool = True) -> None:  # config: str = "config.py"
    """Label the provided report(s)."""
    # delimiter = "."
    # CONFIG_PATH = delimiter.join(["knodle", "labeler", "CheXpert", config])
    # from CONFIG_PATH import *

    # If neg/unc patterns are not in negbio format
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

    # X_matrix = np.array(loader.X_matrix)
    # Aggregate mentions to obtain one set of labels for each report.
    Z_matrix = aggregator.aggregate(loader.collection, extractor.Z_matrix, chexpert_data=chexpert_bool)

    Z_matrix[Z_matrix == UNCERTAIN] = uncertain

    # np.savetxt(os.path.join(OUTPUT_PATH, "X_matrix.csv"), X_matrix, fmt="%s")
    shutil.copy(SAMPLE_PATH, os.path.join(OUTPUT_DIR, "X_matrix.csv"))
    np.savetxt(os.path.join(OUTPUT_DIR, "T_matrix.csv"), loader.T_matrix, delimiter=",")
    np.savetxt(os.path.join(OUTPUT_DIR, "Z_matrix.csv"), Z_matrix, delimiter=",")
    # return X_matrix, loader.T_matrix, Z_matrix

#
# if __name__ == "__main__":
#     parser = ArgParser()
#     label(parser.parse_args(), transform_patterns=False, uncertain=POSITIVE)
