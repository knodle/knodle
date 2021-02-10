import numpy as np
from knodle.trainer.snorkel.utils import z_t_matrix_to_snorkel_matrix


def test_z_t_matrix_to_snorkel_matrix():
    z = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    t = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ])

    snorkel_gold = np.array([
        [-1, 1, -1, -1],
        [-1, -1, 0, -1]
    ])

    snorkel_test = z_t_matrix_to_snorkel_matrix(z, t)
    np.testing.assert_equal(snorkel_gold, snorkel_test)
