import logging

from torch import Tensor, argmax
import numpy as np
from torch.utils.data import TensorDataset


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


def get_majority_vote_probs(
    rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray
):
    """
    This function calculates a majority vote probability for all rule_matches_z. First rule counts will be
    calculated,
    then a probability will be calculated by dividing the values row-wise with the sum. To counteract zero
    division
    all nan values are set to zero.
    Args:
        rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
    Returns:

    """
    rule_counts = np.matmul(rule_matches_z, mapping_rules_labels_t)
    rule_counts_probs = rule_counts / rule_counts.sum(axis=1).reshape(-1, 1)

    rule_counts_probs[np.isnan(rule_counts_probs)] = 0
    return rule_counts_probs


def extract_tensor_from_dataset(dataset: TensorDataset, tensor_index: int) -> Tensor:
    """
    Extracts a tensor from a dataset.
    Args:
        dataset: Dataset to extract tensor from
        tensor_index: Which tensor to extract

    Returns: Tensor

    """
    return dataset.tensors[tensor_index]
