import numpy as np
import pytest

from knodle.transformation.majority import probabilities_to_majority_vote, handle_non_labeled, z_t_matrices_to_probs


def test_z_t_matrices_to_probs():
    z = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1]
        ]
    )
    t = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
        ]
    )
    gold_probs = np.array(
        [
            [1, 0],
            [0, 0],
            [0.5, 0.5]
        ]
    )

    majority_probs = z_t_matrices_to_probs(z, t)
    assert np.array_equal(gold_probs, majority_probs)


def test_probabilities_to_majority_vote_base():
    probs = np.array([0.2, 0, 0, 0.6, 0, 0.2, 0, 0])
    true_label = 3
    label = probabilities_to_majority_vote(probs)
    assert label == true_label


def test_probabilities_to_majority_vote_random():
    probs = np.array([0.5, 0, 0, 0.5, 0, 0, 0.5, 0])
    true_random_labels = [0, 3, 6]
    label = probabilities_to_majority_vote(probs)
    assert label in true_random_labels


def test_probabilities_to_majority_vote_other():
    probs = np.array([0.5, 0, 0, 0.5, 0, 0, 0, 0])
    true_label = 7
    label = probabilities_to_majority_vote(
        probs, choose_random_label=False, choose_other_label=True, other_class_id=7)
    assert label == true_label


def test_probabilities_to_majority_vote_error():
    probs = np.array([0.5, 0, 0, 0.5, 0, 0, 0, 0])
    with pytest.raises(ValueError, match='Specify how to resolve unclear majority votes.'):
        probabilities_to_majority_vote(probs, choose_random_label=False, choose_other_label=False)


def test_handle_non_labeled_error():
    noisy_y_train = np.array(5)
    with pytest.raises(ValueError, match='noisy_y_train needs to be a matrix of dimensions num_samples x num_classes'):
        handle_non_labeled(None, noisy_y_train)


# def test_handle_non_labeled_filter(filter_input):
#     input_dataset, noisy_y_train = filter_input
#
#     gold_ids = np.ones((2, 4))
#     gold_ids[0, 0] = 0
#     gold_mask = np.ones((2, 4))
#     gold_mask[1, 1] = 0
#     gold_probs = np.array([
#         [0.5, 0.5],
#         [0.3, 0.7]
#     ])
#
#     new_input_dataset, new_probs = handle_non_labeled(input_dataset, noisy_y_train, rule_matches_z=None)
#
#     assert np.array_equal(new_input_dataset.tensors[0].detach().numpy(), gold_ids)
#     assert np.array_equal(new_input_dataset.tensors[1].detach().numpy(), gold_mask)
#     assert np.array_equal(new_probs, gold_probs)


def test_handle_non_labeled_other():
    noisy_y_train = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
    y_pred_gold = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0]])

    _, y_pred, _ = handle_non_labeled(
        input_data_x=None,
        noisy_y_train=noisy_y_train,
        rule_matches_z=None,
        filter_non_labelled=False,
        choose_other_label_for_empties=True,
        choose_random_label_for_empties=False,
        preserve_non_labeled_for_empties=False,
        other_class_id=0
    )

    assert np.array_equal(y_pred_gold, y_pred)


def test_handle_non_labeled_random():
    noisy_y_train = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred_gold = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]])
    ]

    _, y_pred, _ = handle_non_labeled(
        input_data_x=None,
        noisy_y_train=noisy_y_train,
        rule_matches_z=None,
        filter_non_labelled=False,
        other_class_id=0,
        choose_other_label_for_empties=False,
        choose_random_label_for_empties=True,
        preserve_non_labeled_for_empties=False,
    )

    # assert np.any(y_pred in y_pred_gold)
    assert np.any(np.all(y_pred == y_pred_gold, axis=1))


def test_handle_non_labeled_preserve():
    noisy_y_train = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])

    _, y_pred, _ = handle_non_labeled(
        input_data_x=None,
        noisy_y_train=noisy_y_train,
        rule_matches_z=None,
        filter_non_labelled=False,
        other_class_id=0,
        choose_other_label_for_empties=False,
        choose_random_label_for_empties=False,
        preserve_non_labeled_for_empties=True,
    )

    # assert np.any(y_pred in y_pred_gold)
    assert np.array_equal(noisy_y_train, y_pred)
