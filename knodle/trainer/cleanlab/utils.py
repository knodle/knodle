import numpy as np
import scipy.sparse as sp

from knodle.transformation.filter import filter_empty_probabilities, filter_probability_threshold
from knodle.transformation.majority import z_t_matrices_to_majority_vote_probs, probabilities_to_majority_vote


def calculate_sample_weights(
        num_classes: int, noise_matrix: np.ndarray, noisy_labels_pruned: np.ndarray, mapping_rules_labels_t: np.ndarray,
        rule_matches_z: np.ndarray
) -> np.ndarray:
    """
    Re-weight examples in the loss function for the final fitting s.t. the "apparent" original number of examples in
    each class is preserved, even though the pruned sets may differ.
    """
    # todo: rewrite!! it is old function from the original CL, it doesn't work here!
    sample_weight = np.ones(np.shape(noisy_labels_pruned))
    for k in range(num_classes):
        noise_matrix_class = 0
        for r in range(noise_matrix.shape[0]):
            if mapping_rules_labels_t[r][k] == 1:
                noise_matrix_class += noise_matrix[r][k]
            else:
                continue
        sample_weight_k = 1.0 / noise_matrix_class
        sample_weight[noisy_labels_pruned == k] = sample_weight_k
    return sample_weight


def calculate_threshold(psx, noisy_y_train, output_classes=None):
    """ calculate thresholds per class: P(we predict the given noisy label is k | given noisy label is k)
    - the same way it is done in the original CL
    """
    if not output_classes:
        output_classes = int(max(noisy_y_train) + 1)

    return np.asarray(
        [np.mean(psx[:, k][np.asarray(noisy_y_train) == k]) for k in range(output_classes)]
    )


def calculate_labels(model_input_x, psx_model_input_x, rule_matches_z, mapping_rules_labels_t, config):
    # todo: mb merge all checks to a separate function (in majority.py as well)?
    if config.filter_non_labelled and config.probability_threshold is not None:
        raise ValueError(
            "You can either filter all non labeled samples or those with probabilities below threshold.")
    if config.other_class_id is not None and config.filter_non_labelled:
        raise ValueError("You can either filter samples with no weak labels or add them to the other class.")

    # calculate labels based on t and z; perform additional filtering if applicable
    labels_probs = z_t_matrices_to_majority_vote_probs(rule_matches_z, mapping_rules_labels_t)

    #  filter out samples where no pattern matched
    if config.filter_non_labelled:
        model_input_x, labels_probs_filtered_, rule_matches_z_filtered_ = filter_empty_probabilities(
            model_input_x, labels_probs, rule_matches_z
        )
        psx_model_input_x, labels_probs_filtered, rule_matches_z_filtered = filter_empty_probabilities(
            psx_model_input_x, labels_probs, rule_matches_z
        )

        assert np.array_equal(labels_probs_filtered, labels_probs_filtered_)

        if isinstance(rule_matches_z_filtered, sp.csr_matrix):
            assert np.sum(rule_matches_z_filtered != rule_matches_z_filtered) == 0
        else:
            assert np.array_equal(rule_matches_z_filtered, rule_matches_z_filtered_)

        labels_probs = labels_probs_filtered
        rule_matches_z = rule_matches_z_filtered

    #  filter out samples where that have probabilities below the threshold
    elif config.probability_threshold is not None:
        model_input_x, labels_probs_filtered_ = filter_probability_threshold(
            model_input_x, labels_probs, probability_threshold=config.probability_threshold
        )
        psx_model_input_x, labels_probs_filtered = filter_probability_threshold(
            psx_model_input_x, labels_probs, probability_threshold=config.probability_threshold
        )
        assert np.array_equal(labels_probs_filtered, labels_probs_filtered_)
        labels_probs = labels_probs_filtered

    kwargs = {
        "choose_random_label": config.choose_random_label,
        "other_class_id": config.other_class_id
    }
    labels = np.apply_along_axis(probabilities_to_majority_vote, axis=1, arr=labels_probs, **kwargs)

    return model_input_x, psx_model_input_x, rule_matches_z, labels


"""
[[0.19840295 0.80159705] 1
 [0.06597774 0.93402226] 1
 [0.13508772 0.86491228] 1
 [0.20564516 0.79435484] 1
 [0.8204965  0.1795035 ] 0
 [0.94812925 0.05187075] 0
 [0.69230769 0.30769231] 0 
 [0.0952381  0.9047619 ] 1
 [0.09722222 0.90277778]] 1
 """