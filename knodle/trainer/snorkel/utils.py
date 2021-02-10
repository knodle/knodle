import numpy as np
import scipy.sparse as sp


def z_t_matrix_to_snorkel_matrix(rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray) -> np.ndarray:
    snorkel_matrix = -1 * np.ones(rule_matches_z.shape)

    if isinstance(rule_matches_z, sp.csr_matrix):
        rule_matches_z = rule_matches_z.toarray()

    z_to_t = np.argmax(mapping_rules_labels_t, axis=-1)
    for i in range(rule_matches_z.shape[0]):
        non_zero_idx = np.where(rule_matches_z[i] != 0)[0]
        snorkel_matrix[i, non_zero_idx] = z_to_t[non_zero_idx]

    return snorkel_matrix