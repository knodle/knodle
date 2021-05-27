import numpy as np

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
        np.array([True, False,  True,  True]),
        np.array([False, True, False, False])]
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
    to_reduce_mask = np.array([1, 0, 0, 1], dtype=np.bool)

    merged_rule_matches_z = _get_merged_matrix(
        full_matches=rule_matches_z, to_reduce_mask=to_reduce_mask, label_rule_masks=expected_iterator)
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
        rule_matches_z=rule_matches_z, mapping_rule_class_t=mapping_rule_class_t,
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
        "mapping_rule_class_t": np.array([
            [1, 0],
            [0, 1]
        ])
    }

    assert np.array_equal(out.get("train_rule_matches_z"), expected["train_rule_matches_z"])
    assert np.array_equal(out.get("test_matches"), expected["test_matches"])
    assert np.array_equal(out.get("mapping_rule_class_t"), expected["mapping_rule_class_t"])

    # test end-to-end by drop
    out = reduce_rule_matches(
        rule_matches_z=rule_matches_z, mapping_rule_class_t=mapping_rule_class_t,
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
        "mapping_rule_class_t": np.array([
            [1, 0],
            [1, 0]
        ])
    }

    assert np.array_equal(out.get("train_rule_matches_z"), expected["train_rule_matches_z"])
    assert np.array_equal(out.get("test_matches"), expected["test_matches"])
    assert np.array_equal(out.get("mapping_rule_class_t"), expected["mapping_rule_class_t"])

