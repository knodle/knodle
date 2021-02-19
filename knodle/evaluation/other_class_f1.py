#!/usr/bin/env python

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import sys
import logging

from collections import Counter
from typing import Dict

import numpy as np

from knodle.transformation.labels import translate_predictions


logger = logging.getLogger(__name__)

def other_class_classification_report(
        y_pred: np.array, y_true: np.array, labels2ids: Dict, verbose: bool = True
) -> Dict:
    string_prediction, string_gold = translate_predictions(
        predictions=y_pred, labels=y_true, labels2ids=labels2ids
    )
    clf_report = score(key=string_gold, prediction=string_prediction, verbose=verbose)
    return clf_report


NO_RELATION = "no_relation"


def score(key, prediction, verbose=False):  # key ist ein batch, prediction auch
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()
    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
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
            line = f"{relation:{longest_relation}s}    P: {prec:6.2f}    R: {recall:6.2f}    F1: {f1:6.2f}    #: {gold:6d}"
            logger.info(line)

    # Print the aggregate score
    if verbose:
        logger.info("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

    logger.info("Precision (micro): {:.3%}".format(prec_micro))
    logger.info("   Recall (micro): {:.3%}".format(recall_micro))
    logger.info("       F1 (micro): {:.3%}".format(f1_micro))

    logger.info("\n")

    return {"precision": prec_micro, "recall": recall_micro, "f1": f1_micro}