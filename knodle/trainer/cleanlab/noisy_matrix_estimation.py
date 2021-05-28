from typing import Union, List, Tuple
import numpy as np
from cleanlab.latent_estimation import estimate_latent, compute_confident_joint, calibrate_confident_joint
from cleanlab.util import value_counts


def calculate_noise_matrix(
        noisy_labels: np.ndarray,
        psx: np.ndarray,
        rule_matches_z: np.ndarray,
        num_classes: int,
        noise_matrix: str = "rule2class",
        calibrate: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[None, None, None, None]]:

    # calculate noise matrix as a (#rules x #classes) matrix - i.e., original noisy inputs are given with correspondence
    # to rules matched in each sample, while the estimated labels are aggregated pro class.
    if noise_matrix == "rule2class":
        return estimate_noise_matrix(
            noisy_labels, psx, rule_matches_z, num_classes, calibrate=calibrate
        )

    # if no special noise matrix calculation method is specified, it will be calculated in CL as usual
    elif noise_matrix == "class2class":
        return None, None, None, None

    else:
        raise ValueError("Unknown noise matrix calculation method.")


def estimate_noise_matrix(
        noisy_labels: np.ndarray,
        psx: np.ndarray,
        rule_matches_z: np.ndarray,
        num_classes: int = None,
        thresholds: List = None,
        converge_latent_estimates: bool = False,
        calibrate: bool = True,
        py_method: str = 'cnt'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    confident_joint = compute_confident_joint_rule2class(
        noisy_labels, psx, rule_matches_z, num_classes, thresholds, calibrate=calibrate)

    # confident_joint = compute_confident_joint(noisy_labels, psx, rule_matches_z, num_classes, thresholds)

    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        s=noisy_labels,
        py_method=py_method,
        converge_latent_estimates=converge_latent_estimates,
    )

    return py, noise_matrix, inv_noise_matrix, confident_joint


def compute_confident_joint_rule2class(
        noisy_labels: np.ndarray,
        psx: np.ndarray,
        rule_matches_z: np.ndarray,
        num_classes: int,
        thresholds: List = None,
        calibrate: bool = True
) -> np.ndarray:
    # todo: add multi_label functionality (see original code, _compute_confident_joint_multi_label function)

    noisy_labels = np.asarray(noisy_labels)

    if thresholds is None:
        # P(we predict the given noisy label is k | given noisy label is k) - the same way it is done in the original CL
        thresholds = np.asarray([np.mean(psx[:, k][noisy_labels == k]) for k in range(num_classes)])

    psx_bool = (psx >= thresholds - 1e-6)
    confident_argmax = psx_bool.argmax(axis=1)  # NB! cases where all probs are below threshold, will receive class 0

    num_confident_bins = psx_bool.sum(axis=1)  # how many class probs are above threshold?
    at_least_one_confident = num_confident_bins > 0
    more_than_one_confident = num_confident_bins > 1

    psx_argmax = psx.argmax(axis=1)
    # confident_argmax if there is one confident/no confident, psx_argmax if there are more than one confident
    true_label_guess = np.where(more_than_one_confident, psx_argmax, confident_argmax)
    # drop samples where there are no confident
    y_confident = true_label_guess[at_least_one_confident]

    # get indices of samples that belong to a particular class
    sample_indices_per_class = [np.where(y_confident == k)[0] for k in range(num_classes)]
    # calculate the total number of rule matched in samples in each class
    rule_matches_per_class = [rule_matches_z[class_sample_indices].sum(axis=0)
                              for class_sample_indices in sample_indices_per_class]
    confident_joint = np.array(rule_matches_per_class).T

    if calibrate:
        confident_joint = calibrate_confident_joint_rule2class(confident_joint, rule_matches_z)

    return confident_joint


def calibrate_confident_joint_rule2class(confident_joint: np.ndarray, rule_matches_z: np.ndarray) -> np.ndarray:

    noisy_labels_counts = rule_matches_z.sum(axis=0)

    calibrated_cj = (confident_joint.T / confident_joint.sum(axis=1) * noisy_labels_counts).T
    calibrated_cj = calibrated_cj / np.sum(calibrated_cj) * sum(noisy_labels_counts)
    return calibrated_cj
