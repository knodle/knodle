import numpy as np
from scipy import sparse as ss

import torch
from torch.utils.data import TensorDataset
from knodle.trainer.snorkel.utils import (
    z_t_matrix_to_snorkel_matrix,
    prepare_empty_rule_matches,
    add_labels_for_empty_examples
)


def test_z_t_matrix_to_snorkel_matrix():
    # test dense case
    z = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 1]
    ])

    t = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ])

    snorkel_gold = np.array([
        [-1, 1, -1, -1],
        [-1, -1, 0,  1]
    ])

    snorkel_test = z_t_matrix_to_snorkel_matrix(z, t)
    np.testing.assert_equal(snorkel_gold, snorkel_test)

    # test sparse case
    z = ss.csr_matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 1]
    ])

    t = ss.csr_matrix([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ])

    snorkel_gold = np.array([
        [-1, 1, -1, -1],
        [-1, -1, 0,  1]
    ])

    snorkel_test = z_t_matrix_to_snorkel_matrix(z, t)
    np.testing.assert_equal(snorkel_gold, snorkel_test)


def test_label_model_data():
    num_samples = 5
    num_rules = 6

    rule_matches_z = np.ones((num_samples, num_rules))
    rule_matches_z[[1, 4]] = 0

    non_zero_mask, out_rule_matches_z = prepare_empty_rule_matches(rule_matches_z)

    expected_mask = np.array([True, False, True, True, False])
    expected_rule_matches = np.ones((3, num_rules))

    np.testing.assert_equal(non_zero_mask, expected_mask)
    np.testing.assert_equal(out_rule_matches_z, expected_rule_matches)


def test_other_class_labels():
    label_probs_gen = np.array([
        [0.3, 0.6, 0.0, 0.1],
        [0.2, 0.2, 0.2, 0.4],
        [1.0, 0.0, 0.0, 0.0]
    ])
    output_classes = 5
    other_class_id = 4

    # test without empty rows
    non_zero_mask = np.array([True, True, True])
    expected_probs = np.array([
        [0.3, 0.6, 0.0, 0.1, 0.0],
        [0.2, 0.2, 0.2, 0.4, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0]
    ])
    label_probs = add_labels_for_empty_examples(label_probs_gen, non_zero_mask, output_classes, other_class_id)

    np.testing.assert_equal(label_probs, expected_probs)

    # test with empty rows
    non_zero_mask = np.array([True, False, False, True, True])
    expected_probs = np.array([
        [0.3, 0.6, 0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.2, 0.2, 0.2, 0.4, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0]
    ])
    label_probs = add_labels_for_empty_examples(label_probs_gen, non_zero_mask, output_classes, other_class_id)

    np.testing.assert_equal(label_probs, expected_probs)
