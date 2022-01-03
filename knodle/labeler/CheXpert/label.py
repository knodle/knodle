"""Entry-point script to label radiology reports."""
from args import ArgParser
from stages import Loader, Extractor, Classifier, Aggregator, transform
from knodle.labeler.CheXpert.stages.constants import *


def label(args, transform_patterns=False):
    """Label the provided report(s)."""

    # If neg/unc patterns are not in negbio format
    if transform_patterns:
        transform(args.pre_negation_uncertainty_path)
        transform(args.negation_path)
        transform(args.post_negation_uncertainty_path)

    loader = Loader(args.reports_path, args.extract_impression)

    extractor = Extractor(args.mention_phrases_dir,
                          args.unmention_phrases_dir,
                          verbose=args.verbose)
    classifier = Classifier(args.pre_negation_uncertainty_path,
                            args.negation_path,
                            args.post_negation_uncertainty_path,
                            verbose=args.verbose)
    aggregator = Aggregator(CATEGORIES,
                            verbose=args.verbose)

    # Load reports in place.
    loader.load()
    # Extract observation mentions in place.
    extractor.extract(loader.collection)
    # Classify mentions in place.
    classifier.classify(loader.collection)

    X_matrix = loader.reports.to_numpy()
    # Aggregate mentions to obtain one set of labels for each report.
    Z_matrix = aggregator.aggregate(loader.collection, extractor.Z_matrix)

    return X_matrix, loader.T_matrix, Z_matrix

    #write(loader.reports, labels, args.output_path, args.verbose)


if __name__ == "__main__":
    parser = ArgParser()
    label(parser.parse_args())
