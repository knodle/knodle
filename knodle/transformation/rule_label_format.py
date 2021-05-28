import numpy as np


def transform_snorkel_matrix_to_z_t(class_matrix: np.ndarray) -> [np.ndarray, np.ndarray]:
    """Takes a matrix in format used by e.g. Snorkel (https://github.com/snorkel-team/snorkel)
    and transforms it to z / t matrices. Format
        - class_matrix_ij = -1, iff the rule doesn't apply
        - class_matrix_ij = k, iff the rule labels class k

    :param class_matrix: shape=(num_samples, num_weak_labellers)
    :return: Z matrix - binary encoded array of which rules matched. Shape: instances x rules.
             T matrix - mapping of rules to labels, binary encoded. Shape: rules x classes.
    """
    num_classes = class_matrix.max() + 1

    z_matrix = np.where(class_matrix != -1, 1, class_matrix)
    z_matrix = np.where(z_matrix == -1, 0, z_matrix)

    t_matrix = np.zeros((class_matrix.shape[1], num_classes))

    for i in range(class_matrix.shape[1]):
        row = class_matrix[:, i].tolist()
        label = max(row)
        if set(row) != {-1, label}:
            raise RuntimeError(
                "A weak labeller is not allowed to label more than one class."
            )

        t_matrix[i, label] = 1

    return z_matrix, t_matrix
