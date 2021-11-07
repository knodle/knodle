import logging
import numpy as np
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def update_t_matrix_with_prior(
        prune_count_matrix: np.ndarray, t_matrix: np.ndarray, verbose: bool = True
) -> np.ndarray:
    """
    :param prune_count_matrix: C matrix (rules x classes) with confident estimations
    :return:
    """
    prior = np.mean(prune_count_matrix)
    logger.info(f"Prior: {prior}")

    # logger.info("For current experiment with IMDB is temporally increased twice")
    # prior *= 2
    # logger.info(f"New prior is: {prior}")

    t_matrix_with_prior = t_matrix * prior
    updated_t_matrix = normalize(t_matrix_with_prior + prune_count_matrix, axis=1, norm='l1')

    if verbose:
        logger.info(updated_t_matrix)

    return updated_t_matrix


def update_t_matrix(prune_count_matrix, t_matrix, p: float = 0.5, verbose: bool = True) -> np.ndarray:
    normalized_prune_counts = normalize(prune_count_matrix, axis=1, norm='l1')
    updated_t_matrix = t_matrix * (1-p) + normalized_prune_counts * p

    # the first version: sum and divide
    # t_matrix_updated = np.zeros_like(t_matrix, dtype="float")
    # for rule in range(prune_count_matrix.shape[0]):
    #     rule_count_sum = sum(prune_count_matrix[rule, :]) + sum(t_matrix[rule, :])
    #     for label in range(prune_count_matrix.shape[1]):
    #         t_matrix_updated[rule][label] = (prune_count_matrix[rule][label] + t_matrix[rule][label]) / rule_count_sum

    if verbose:
        logger.info(updated_t_matrix)
    return updated_t_matrix
