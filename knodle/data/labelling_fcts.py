import numpy as np


def transform_lambda_matrix_to_z_t(lambda_matrix: np.array) -> [np.array, np.array]:
    """Takes a matrix in the "lambda format" and transforms it to z / t matrices.
    Lambda format:
        - lambda_ij = -1, iff the rule doesn't apply
        - lambda_ij = k, iff the rule labels class k

    :param lambda_matrix: shape=(num_samples, num_weak_labellers)
    :return: Z, T matrix as described in the paper.
    """
    num_classes = lambda_matrix.max() + 1

    z_matrix = np.where(lambda_matrix != -1, 1, lambda_matrix)
    z_matrix = np.where(z_matrix == -1, 0, z_matrix)

    t_matrix = np.zeros((lambda_matrix.shape[1], num_classes))

    for i in range(lambda_matrix.shape[1]):
        row = lambda_matrix[:, i].tolist()
        label = max(row)
        if set(row) != {-1, label}:
            raise RuntimeError("A weak labeller is not allowed to label more than one class.")

        t_matrix[i, label] = 1

    return z_matrix, t_matrix
