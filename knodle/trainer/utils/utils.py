import logging

from torch import Tensor, argmax


def log_section(text: str, logger: logging) -> None:
    """
    Prints a section
    Args:
        text: Text to print
        logger: Logger object

    Returns:

    """
    logger.info("======================================")
    logger.info(text)
    logger.info("======================================")


def accuracy_of_probs(predictions: Tensor, ground_truth: Tensor):
    """
    Function to calculate the accuracy of two tensors with probabilities attached.
    Args:
        predictions: Predictions, shape: instances x labels
        ground_truth: Ground truth, shape one of (instaces x 1) or (instaces x labels)

    Returns: Accuracy

    """
    ground_truth = (
        ground_truth if len(ground_truth.shape) == 1 else argmax(ground_truth, dim=-1)
    )
    y_pred_max = argmax(predictions, dim=-1)
    correct_pred = (y_pred_max == ground_truth).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc
