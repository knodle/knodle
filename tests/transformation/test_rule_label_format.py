import numpy as np

from knodle.transformation.rule_label_format import transform_snorkel_matrix_to_z_t


def test_transform_snorkel_matrix_to_z_t():

    lambda_matrix = np.array([[-1, 2, -1, 1], [1, -1, -1, 1], [1, 2, 0, -1]])

    correct_z_matrix = np.array([[0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0]])

    correct_t_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])

    z, t = transform_snorkel_matrix_to_z_t(lambda_matrix)

    np.array_equal(correct_z_matrix, z)
    np.array_equal(correct_t_matrix, t)
