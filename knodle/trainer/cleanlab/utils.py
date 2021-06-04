import numpy as np


def calculate_sample_weights(num_classes, noise_matrix, noisy_labels_pruned):
    sample_weight = np.ones(np.shape(noisy_labels_pruned))
    for k in range(num_classes):
        sample_weight_k = 1.0 / noise_matrix[k][k]
        sample_weight[noisy_labels_pruned == k] = sample_weight_k
    return sample_weight
