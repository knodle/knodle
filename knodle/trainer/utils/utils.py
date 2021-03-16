import random
import logging

import torch
from torch import Tensor, argmax
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt


def log_section(text: str, logger: logging, additional_info: {} = None) -> None:
    """
    Prints a section
    Args:
        text: Text to print
        logger: Logger object

    Returns:

    """
    logger.info("======================================")
    logger.info(text)
    if additional_info:
        for key, value in additional_info.items():
            logger.info("{}: {}".format(key, value))
    logger.info("======================================")


def accuracy_of_probs(predictions: Tensor, ground_truth: Tensor):
    """
    Function to calculate the accuracy of two tensors with probabilities attached.
    Args:
        predictions: Predictions, shape: instances x labels
        ground_truth: Ground truth, shape one of (instances x 1) or (instances x labels)

    Returns: Accuracy

    """
    ground_truth = (
        ground_truth if len(ground_truth.shape) == 1 else argmax(ground_truth, dim=-1)
    )
    y_pred_max = argmax(predictions, dim=-1)
    correct_pred = (y_pred_max == ground_truth).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc


def extract_tensor_from_dataset(dataset: TensorDataset, tensor_index: int) -> Tensor:
    """
    Extracts a tensor from a dataset.
    Args:
        dataset: Dataset to extract tensor from
        tensor_index: Which tensor to extract

    Returns: Tensor

    """
    return dataset.tensors[tensor_index]


def check_and_return_device() -> torch.device:
    """
    Function to check if a GPU is available and sets the device for pytorch.
    If a GPU is available -> The device is GPU, if not the device is CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def set_seed(seed: int) -> None:
    """ Fix seed for all shuffle processes in order to get the reproducible result """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
