"""Entry-point script to label radiology reports."""
import shutil
import os
import sys
sys.path.append(os.getcwd())
from args import ArgParser
from stages import Loader, Extractor, Classifier, Aggregator, transform
from examples.labeler.chexpert.constants.constants import *


def label(args, transform_patterns=False, uncertain=POSITIVE):  # should uncertain matches be POSITIVE or NEGATIVE
    """Label the provided report(s)."""

    # If neg/unc patterns are not in negbio format
    if transform_patterns:
        transform(PRE_NEG_UNC_PATH)
        transform(NEG_PATH)
        transform(POST_NEG_UNC_PATH)

    loader = Loader()

    extractor = Extractor(verbose=args.verbose)
    classifier = Classifier(verbose=args.verbose)
    aggregator = Aggregator(verbose=args.verbose)

    # Load reports in place.
    loader.load()
    # Extract observation mentions in place.
    extractor.extract(loader.collection)
    # Classify mentions in place.
    classifier.classify(loader.collection)

    # X_matrix = np.array(loader.X_matrix)
    # Aggregate mentions to obtain one set of labels for each report.
    Z_matrix = aggregator.aggregate(loader.collection, extractor.Z_matrix)

    Z_matrix[Z_matrix == UNCERTAIN] = uncertain

    # np.savetxt(os.path.join(OUTPUT_PATH, "X_matrix.csv"), X_matrix, fmt="%s")
    shutil.copy(REPORTS_PATH, os.path.join(OUTPUT_PATH, "X_matrix.csv"))
    np.savetxt(os.path.join(OUTPUT_PATH, "T_matrix.csv"), loader.T_matrix, delimiter=",")
    np.savetxt(os.path.join(OUTPUT_PATH, "Z_matrix.csv"), Z_matrix, delimiter=",")
    # return X_matrix, loader.T_matrix, Z_matrix


if __name__ == "__main__":
    parser = ArgParser()
    label(parser.parse_args(), transform_patterns=False, uncertain=POSITIVE)
