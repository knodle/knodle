import logging

from collections import Counter
from typing import Dict, List

import numpy as np

from knodle.transformation.labels import label_ids_to_labels

logger = logging.getLogger(__name__)


def classification_report_other_class(
        y_true: np.array, y_pred: np.array, ids2labels: Dict, other_class_id: int, verbose: bool = True
) -> Dict:
    """ Prepare the batch for the label-based evaluation. """
    string_prediction, string_gold = label_ids_to_labels(
        predictions=y_pred, labels=y_true, ids2labels=ids2labels
    )
    clf_report = score(
        key=string_gold, prediction=string_prediction, verbose=verbose, other_class_label=ids2labels[other_class_id]
    )
    return clf_report


def score(
        key: List[str], prediction: List[str], verbose: bool, other_class_label: str
) -> Dict[str, float]:
    """
    Computes the precision, recall and f1-score with respect to the other_class label.
    Other class is treated as a separate negative class not included into the evaluation,
    i.e. a correctly predicted "other_class" label counts as a TN rather than a TP.
    Args:
        key: List with gold string labels for the batch
        prediction: List with predicted labels for the batch
        verbose: Whether to output per-relation statistics
        other_class_label: Label of the class that should be treated as negative.
    Returns: Decision per sample. Shape: (instances, )
    """
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()
    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]
        if gold == other_class_label and guess == other_class_label:
            pass
        elif gold == other_class_label and guess != other_class_label:
            guessed_by_relation[guess] += 1
        elif gold != other_class_label and guess == other_class_label:
            gold_by_relation[gold] += 1
        elif gold != other_class_label and guess != other_class_label:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        logger.info("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            line = f"{relation:{longest_relation}s}\tP: {prec:6.2f}    R: {recall:6.2f}    " \
                f"F1: {f1:6.2f}    #: {gold:6d}"
            logger.info(line)

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

    # Print the aggregate score
    logger.info("Final Score:")
    logger.info("Precision (micro): {:.3%}".format(prec_micro))
    logger.info("   Recall (micro): {:.3%}".format(recall_micro))
    logger.info("       F1 (micro): {:.3%}".format(f1_micro))
    logger.info("\n")

    return {"precision": prec_micro, "recall": recall_micro, "f1": f1_micro}
