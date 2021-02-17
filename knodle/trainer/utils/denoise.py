import numpy as np


def activate_neighbors(
        rule_matches_z: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """
    Find all closest neighbors and take the same label ids
    Args:
        rule_matches_z: All rule matches. Shape: instances x rules
        indices: Neighbor indices from knn
    Returns:
    """
    new_z_matrix = np.zeros(rule_matches_z.shape)
    for index, sample in enumerate(rule_matches_z):
        neighbors = indices[index].astype(int)
        neighborhood_z = rule_matches_z[neighbors, :]

        activated_lfs = neighborhood_z.sum(axis=0)  # Add all lf activations

        # All with != 0 are valid. We could also do some division here for weighted
        activated_lfs = np.where(activated_lfs > 0)[0]
        new_z_matrix[index, activated_lfs] = 1

    return new_z_matrix
