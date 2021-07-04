import numpy as np


def calculate_sample_weights(
        num_classes: int, noise_matrix: np.ndarray, noisy_labels_pruned: np.ndarray, mapping_rules_labels_t: np.ndarray,
        rule_matches_z: np.ndarray
) -> np.ndarray:
    """
    Re-weight examples in the loss function for the final fitting s.t. the "apparent" original number of examples in
    each class is preserved, even though the pruned sets may differ.
    """
    # todo: rewrite!! it is old function from the original CL, it doesn't work here!
    sample_weight = np.ones(np.shape(noisy_labels_pruned))
    for k in range(num_classes):
        noise_matrix_class = 0
        for r in range(noise_matrix.shape[0]):
            if mapping_rules_labels_t[r][k] == 1:
                noise_matrix_class += noise_matrix[r][k]
            else:
                continue
        sample_weight_k = 1.0 / noise_matrix_class
        sample_weight[noisy_labels_pruned == k] = sample_weight_k
    return sample_weight
