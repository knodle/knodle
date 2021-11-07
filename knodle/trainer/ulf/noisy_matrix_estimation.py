import logging
from typing import Union, Tuple
import numpy as np
import scipy.sparse as sp
from cleanlab.util import clip_values

logger = logging.getLogger(__name__)


def calculate_noise_matrix(
        psx: np.ndarray,
        rule_matches_z: np.ndarray,
        thresholds: np.ndarray,
        num_classes: int,
        noise_matrix: str = "rule2class",
        calibrate: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[None, None, None]]:

    # calculate noise matrix as a (#rules x #classes) matrix - i.e., original noisy inputs are given with correspondence
    # to rules matched in each sample, while the estimated labels are aggregated pro class.
    if noise_matrix == "rule2class":
        return estimate_noise_matrix_rule2class(psx, rule_matches_z, thresholds, num_classes, calibrate=calibrate)

    # if no special noise matrix calculation method is specified, it will be calculated in CL as usual
    elif noise_matrix == "class2class":
        return None, None, None

    else:
        raise ValueError("Unknown noise matrix calculation method.")


def estimate_noise_matrix_rule2class(
        psx: np.ndarray,
        rule_matches_z: np.ndarray,
        thresholds: np.ndarray,
        num_classes: int,
        calibrate: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    confident_joint = compute_confident_joint_rule2class(psx, rule_matches_z, thresholds, num_classes, calibrate)

    _, noise_matrix, inv_noise_matrix = estimate_latent_rule2class(
        confident_joint=confident_joint,
        rule_matches_z=rule_matches_z,
        # py_method=py_method,
        # converge_latent_estimates=converge_latent_estimates,
    )

    return noise_matrix, inv_noise_matrix, confident_joint


def compute_confident_joint_rule2class(
        psx: np.ndarray,
        rule_matches_z: np.ndarray,
        thresholds: np.ndarray,
        num_classes: int,
        calibrate: bool = True
) -> np.ndarray:
    # todo: add multi_label functionality (see original code, _compute_confident_joint_multi_label function)

    # boolean matrix, where True means the probability is above threshold for this class, False - below.
    psx_bool = (psx >= thresholds - 1e-6)

    # confident_argmax: 1 if class 1 is above the threshold, 0 if class 0 is above the threshold OR probs for all
    # classes are below threshold
    confident_argmax = psx_bool.argmax(axis=1)

    # num_confident_bins: number of classes above threshold
    num_confident_bins = np.nansum(psx_bool, axis=1)
    # at_least_one_confident: boolean; True if only one class is above threshold
    at_least_one_confident = num_confident_bins > 0
    # more_than_one_confident: boolean; True if more than one class is above threshold
    more_than_one_confident = num_confident_bins > 1

    # psx_argmax: predicted labels (= the class that got the highest probability)
    psx_argmax = psx.argmax(axis=1)
    # if there is one confident/no confident: confident_argmax, if there are more than one confident: psx_argmax
    true_label_guess = np.where(more_than_one_confident, psx_argmax, confident_argmax)
    # replace samples where there are no confident to -1 (in order to disregard them in further calculations)
    y_confident = np.copy(true_label_guess)
    y_confident[~at_least_one_confident] = -1

    # get indices of samples that belong to a particular class
    sample_indices_per_class = [np.where(y_confident == k)[0] for k in range(num_classes)]
    # calculate the number of each rule matched in samples in each class
    if isinstance(rule_matches_z, sp.csr_matrix):
        rule_matches_per_class = [
            np.squeeze(np.nansum(rule_matches_z[sample_idx, :], axis=0)) for sample_idx in sample_indices_per_class
        ]
    else:
        rule_matches_per_class = [
            np.nansum(rule_matches_z[sample_idx], axis=0) for sample_idx in sample_indices_per_class
            # rule_matches_z[sample_idx].nansum(axis=0) for sample_idx in sample_indices_per_class
        ]
    confident_joint = np.array(rule_matches_per_class).T

    if calibrate:
        return calibrate_confident_joint_rule2class(confident_joint, rule_matches_z)

    return confident_joint


def estimate_latent_rule2class(
        confident_joint: np.ndarray, rule_matches_z: np.ndarray, py_method: str = 'cnt'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the latent prior p(y), the noise matrix P(LFs|y) and the inverse noise matrix P(y|LFs) from the
    `confident_joint` count(LFs, y). The `confident_joint` estimated by `compute_confident_joint` by counting confident
    examples.
    :return:
    """
    # 'p_rules' is p(rule)
    p_rules = np.sum(rule_matches_z, axis=0) / rule_matches_z.shape[0]
    # Number of training examples confidently counted for each rule
    rule_count = confident_joint.sum(axis=1).astype(float)
    # Number of training examples confidently counted into each true (= defined by cross-validation) class
    y_count = confident_joint.sum(axis=0).astype(float)
    # Confident Counts Estimator: p(s=k_s|y=k_y) ~ |s=k_s and y=k_y| / |y=k_y|
    noise_matrix = confident_joint / y_count
    # Confident Counts Estimator: p(y=k_y|s=k_s) ~ |y=k_y and s=k_s| / |s=k_s|
    inv_noise_matrix = confident_joint.T / rule_count
    # todo: add clip_noise_rates from the original CL estimate_latent ?
    # py = compute_py_rule2class(p_rules, noise_matrix, inv_noise_matrix, py_method)
    py = 0

    # todo: add converge_latent_estimates
    return py, noise_matrix, inv_noise_matrix


def calibrate_confident_joint_rule2class(confident_joint: np.ndarray, rule_matches_z: np.ndarray) -> np.ndarray:
    if isinstance(rule_matches_z, sp.csr_matrix):
        sample_pro_rule_counts_ = np.squeeze(np.array(np.nansum(rule_matches_z, axis=0)))
        calibrated_cj = (confident_joint.T / np.nansum(confident_joint, axis=1) * sample_pro_rule_counts_).T
        return np.nan_to_num(calibrated_cj)
    else:
        sample_pro_rule_counts = np.nansum(rule_matches_z, axis=0)
        # calibrated_cj = calibrated_cj / np.sum(calibrated_cj) * sum(noisy_labels_counts)        # double check
        calibrated_cj = (confident_joint.T / np.nansum(confident_joint, axis=1) * sample_pro_rule_counts).T
        return np.nan_to_num(calibrated_cj)


def compute_py_rule2class(ps, noise_matrix, inverse_noise_matrix, py_method='cnt'):
    if len(np.shape(ps)) > 2 or (
            len(np.shape(ps)) == 2 and np.shape(ps)[0] != 1):
        w = 'Input parameter np.array ps has shape ' + str(np.shape(ps))
        w += ', but shape should be (K, ) or (1, K)'
        logger.warning.warn(w)

    if py_method != 'cnt':
        logger.info("The only supported calculation method of the latent prior p(y=k) is now cnt, so it will be used")

    # Computing py this way avoids dividing by zero noise rates.
    # More robust bc error est_p(y|s) / est_p(s|y) ~ p(y|s) / p(s|y)
    py = inverse_noise_matrix.diagonal() / noise_matrix.diagonal() * ps
    # Equivalently,
    # py = (y_count / s_count) * ps

    # Clip py (0,1), .s.t. no class should have prob 0, hence 1e-5
    py = clip_values(py, low=1e-5, high=1.0, new_sum=1.0)
    return py
