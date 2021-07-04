import numpy as np
from scipy.sparse import csr_matrix

from knodle.transformation.rule_reduction import _get_merged_matrix, reduce_rule_matches, _get_rule_by_label_iterator


def test_reduction():
    # test rule iterator
    mapping_rule_class_t = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [1, 0]])
    rule_iterator = list(_get_rule_by_label_iterator(mapping_rule_class_t))
    expected_iterator = [
        np.array([0, 2, 3]),
        np.array([1])
    ]

    assert len(rule_iterator) == len(expected_iterator)
    assert np.array_equal(rule_iterator[0], expected_iterator[0])
    assert np.array_equal(rule_iterator[1], expected_iterator[1])

    # test _get_merged_matrix
    rule_matches_z = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 0]])
    to_keep_mask = np.array([1, 2])

    merged_rule_matches_z = _get_merged_matrix(
        full_matches=rule_matches_z, to_keep_mask=to_keep_mask, label_rule_masks=expected_iterator)
    expected_merged = np.array([
        [1],
        [1],
        [0],
        [1],
        [0]])
    assert np.array_equal(merged_rule_matches_z, expected_merged)

    # test end-to-end merge
    test_rule_matches_z = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0]])

    out = reduce_rule_matches(
        rule_matches_z=rule_matches_z, mapping_rules_labels_t=mapping_rule_class_t,
        rule_matches_rest={"test_matches": test_rule_matches_z},
        drop_rules=False, max_rules=2, min_coverage=1.0)

    expected = {
        "train_rule_matches_z": np.array([
            [1, 1],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0]
        ]),
        "test_matches": np.array([
            [0, 1],
            [1, 0],
            [1, 0]
        ]),
        "mapping_rules_labels_t": np.array([
            [1, 0],
            [0, 1]
        ])
    }

    assert np.array_equal(out.get("train_rule_matches_z"), expected["train_rule_matches_z"])
    assert np.array_equal(out.get("test_matches"), expected["test_matches"])
    assert np.array_equal(out.get("mapping_rules_labels_t"), expected["mapping_rules_labels_t"])

    # test end-to-end by drop
    out = reduce_rule_matches(
        rule_matches_z=rule_matches_z, mapping_rules_labels_t=mapping_rule_class_t,
        rule_matches_rest={"test_matches": test_rule_matches_z},
        drop_rules=True, max_rules=2, min_coverage=0.0)

    expected = {
        "train_rule_matches_z": np.array([
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 1],
            [0, 1]
        ]),
        "test_matches": np.array([
            [0, 0],
            [0, 1],
            [1, 0]
        ]),
        "mapping_rules_labels_t": np.array([
            [1, 0],
            [1, 0]
        ])
    }

    assert np.array_equal(out.get("train_rule_matches_z"), expected["train_rule_matches_z"])
    assert np.array_equal(out.get("test_matches"), expected["test_matches"])
    assert np.array_equal(out.get("mapping_rules_labels_t"), expected["mapping_rules_labels_t"])


def test_reduction_for_sparse():
    # test rule iterator
    mapping_rule_class_t = csr_matrix([
        [1, 0],
        [0, 1],
        [1, 0],
        [1, 0]])
    rule_iterator = list(_get_rule_by_label_iterator(mapping_rule_class_t))
    expected_iterator = [
        np.array([0, 2, 3]),
        np.array([1])
    ]

    assert len(rule_iterator) == len(expected_iterator)
    assert np.array_equal(rule_iterator[0], expected_iterator[0])
    assert np.array_equal(rule_iterator[1], expected_iterator[1])

    # test _get_merged_matrix
    rule_matches_z = csr_matrix([
        [0, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 0]])
    to_keep_mask = np.array([1, 2])

    merged_rule_matches_z = _get_merged_matrix(
        full_matches=rule_matches_z, to_keep_mask=to_keep_mask, label_rule_masks=expected_iterator)
    expected_merged = csr_matrix([
        [1],
        [1],
        [0],
        [1],
        [0]])
    assert (merged_rule_matches_z != expected_merged).nnz == 0

    # test end-to-end merge
    test_rule_matches_z = csr_matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0]])

    out = reduce_rule_matches(
        rule_matches_z=rule_matches_z, mapping_rules_labels_t=mapping_rule_class_t,
        rule_matches_rest={"test_matches": test_rule_matches_z},
        drop_rules=False, max_rules=2, min_coverage=1.0)

    expected = {
        "train_rule_matches_z": csr_matrix([
            [1, 1],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0]
        ]),
        "test_matches": csr_matrix([
            [0, 1],
            [1, 0],
            [1, 0]
        ]),
        "mapping_rules_labels_t": csr_matrix([
            [1, 0],
            [0, 1]
        ])
    }

    assert (out.get("train_rule_matches_z") != expected["train_rule_matches_z"]).nnz == 0
    assert (out.get("test_matches") != expected["test_matches"]).nnz == 0
    assert (out.get("mapping_rules_labels_t") != expected["mapping_rules_labels_t"]).nnz == 0
    assert isinstance(out.get("train_rule_matches_z"), csr_matrix)
    assert isinstance(out.get("mapping_rules_labels_t"), csr_matrix)

    # test end-to-end by drop with sparse matches and dense mapping T
    mapping_rule_class_t = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [1, 0]])

    out = reduce_rule_matches(
        rule_matches_z=rule_matches_z, mapping_rules_labels_t=mapping_rule_class_t,
        rule_matches_rest={"test_matches": test_rule_matches_z},
        drop_rules=True, max_rules=2, min_coverage=0.0)

    expected = {
        "train_rule_matches_z": csr_matrix([
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 1],
            [0, 1]
        ]),
        "test_matches": csr_matrix([
            [0, 0],
            [0, 1],
            [1, 0]
        ]),
        "mapping_rules_labels_t": np.array([
            [1, 0],
            [1, 0]
        ])
    }

    assert (out.get("train_rule_matches_z") != expected["train_rule_matches_z"]).nnz == 0
    assert (out.get("test_matches") != expected["test_matches"]).nnz == 0
    assert np.array_equal(out.get("mapping_rules_labels_t"), expected["mapping_rules_labels_t"])
    assert isinstance(out.get("train_rule_matches_z"), csr_matrix)
    assert isinstance(out.get("mapping_rules_labels_t"), np.ndarray)

