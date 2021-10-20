
# coding: utf-8

# The LearningWithNoisyLabelsTorch algorithm class for multiclass learning with noisy labels.
# The LearningWithNoisyLabels inherits a lot of functions
# from LearningWithNoisyLabels in the original Cleanlab library,
# but adapts it for training the PyTorch model.
import multiprocessing
import numpy as np
from torch import nn

from knodle.model.logistic_regression_model import LogisticRegressionModel


class LearningWithNoisyLabelsTorch(nn.Module):

    def __init__(
            self,
            clf=None,
            seed=None,
            # Hyper-parameters
            cv_n_folds=5,
            converge_latent_estimates=False,
            pulearning=None,
            n_jobs=None,
    ):

        super(LearningWithNoisyLabelsTorch, self).__init__()

        if clf is None:
            # Use logistic regression if no classifier is provided.
            clf = LogisticRegressionModel()         # todo: think about initialization (num classes etc)
        if seed is not None:
            np.random.seed(seed=seed)

        # Set-up number of multiprocessing threads used by get_noise_indices()
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()
        else:
            assert (n_jobs >= 1)

        self.clf = clf
        self.seed = seed
        self.cv_n_folds = cv_n_folds
        self.converge_latent_estimates = converge_latent_estimates
        self.pulearning = pulearning
        self.n_jobs = n_jobs
        self.noise_mask = None
        self.sample_weight = None
        self.confident_joint = None
        self.py = None
        self.ps = None
        self.K = None
        self.noise_matrix = None
        self.inverse_noise_matrix = None
