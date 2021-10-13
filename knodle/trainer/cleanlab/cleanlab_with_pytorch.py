import logging
from typing import Union, Tuple

import torch
import numpy as np
import scipy.sparse as sp
from cleanlab.latent_estimation import estimate_py_noise_matrices_and_cv_pred_proba
from cleanlab.pruning import get_noise_indices
from cleanlab.util import value_counts

from torch.utils.data import TensorDataset
from torch.utils.data.dataset import Subset

from knodle.trainer import MajorityVoteTrainer
from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.cleanlab.classification import LearningWithNoisyLabelsTorch
from knodle.trainer.cleanlab.config import CleanLabConfig
from knodle.trainer.cleanlab.pruning import update_t_matrix, update_t_matrix_with_prior
from knodle.trainer.cleanlab.psx_estimation import calculate_psx, estimate_py_noise_matrices_and_cv_pred_proba_baseline
from knodle.trainer.cleanlab.noisy_matrix_estimation import calculate_noise_matrix
from knodle.trainer.cleanlab.utils import calculate_threshold, calculate_labels
from knodle.trainer.wscrossweigh.data_splitting_by_rules import get_dataset_by_sample_ids
from knodle.transformation.filter import filter_empty_probabilities, filter_probability_threshold
from knodle.transformation.majority import probabilities_to_majority_vote, z_t_matrices_to_majority_vote_probs, \
    input_to_majority_vote_input
from knodle.transformation.torch_input import input_info_labels_to_tensordataset

logger = logging.getLogger(__name__)


@AutoTrainer.register('cleanlab')
class CleanLabPyTorchTrainer(MajorityVoteTrainer):

    def __init__(
            self,
            psx_model=None,
            psx_model_input_x=None,
            **kwargs):

        self.psx_model = psx_model if psx_model else kwargs.get("model")
        self.psx_model_input_x = psx_model_input_x if psx_model_input_x else kwargs.get("model_input_x")

        if kwargs.get("trainer_config", None) is None:
            kwargs["trainer_config"] = CleanLabConfig()
        super().__init__(**kwargs)

        if self.trainer_config.use_probabilistic_labels:
            logger.warning("WSCleanlab denoising method is not compatible with probabilistic labels. The labels for "
                           "each sample will be chosen with majority voting instead")
            self.trainer_config.use_probabilistic_labels = False

        self.K = None
        self.ps = None
        self.py = None
        self.noise_matrix = None
        self.inverse_noise_matrix = None
        self.confident_joint = None
        self.noise_mask = None
        self.sample_weight = None

        self.converge_latent_estimates = False      # todo: try with true
        self.pulearning = None
        self.n_jobs = None

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ) -> None:

        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)
        self.trainer_config.optimizer = self.initialise_optimizer()

        self.model_input_x, self.psx_model_input_x, self.rule_matches_z, noisy_y_train = calculate_labels(
            self.model_input_x, self.psx_model_input_x, self.rule_matches_z, self.mapping_rules_labels_t,
            self.trainer_config
        )

        # Number of classes
        self.K = len(np.unique(noisy_y_train))

        # 'ps' is p(s=k)
        self.ps = value_counts(noisy_y_train) / float(len(noisy_y_train))

        self.py, self.noise_matrix, self.inverse_noise_matrix, self.confident_joint, psx = \
            estimate_py_noise_matrices_and_cv_pred_proba_baseline(
                data_features=self.psx_model_input_x,
                labels=noisy_y_train,
                model=self.psx_model,
                config=self.trainer_config,
                cv_n_folds=self.trainer_config.cv_n_folds,
                thresholds=None,
                converge_latent_estimates=(
                    self.converge_latent_estimates),
                seed=self.trainer_config.seed,
            )

        # if pulearning == the integer specifying the class without noise.
        if self.K == 2 and self.pulearning is not None:  # pragma: no cover
            # pulearning = 1 (no error in 1 class) implies p(s=1|y=0) = 0
            self.noise_matrix[self.pulearning][1 - self.pulearning] = 0
            self.noise_matrix[1 - self.pulearning][1 - self.pulearning] = 1
            # pulearning = 1 (no error in 1 class) implies p(y=0|s=1) = 0
            self.inverse_noise_matrix[1 - self.pulearning][self.pulearning] = 0
            self.inverse_noise_matrix[self.pulearning][self.pulearning] = 1
            # pulearning = 1 (no error in 1 class) implies p(s=1,y=0) = 0
            self.confident_joint[self.pulearning][1 - self.pulearning] = 0
            self.confident_joint[1 - self.pulearning][1 - self.pulearning] = 1

        # This is the actual work of this function.

        # Get the indices of the examples we wish to prune
        self.noise_mask = get_noise_indices(
            noisy_y_train,
            psx,
            inverse_noise_matrix=self.inverse_noise_matrix,
            confident_joint=self.confident_joint,
            prune_method=self.trainer_config.prune_method,
            n_jobs=self.n_jobs,
        )

        x_mask = ~self.noise_mask

        s_pruned = noisy_y_train[x_mask]
        x_pruned = TensorDataset(*[inp[x_mask] for inp in self.model_input_x.tensors])

        # x_pruned = self.model_input_x[x_mask]
        # s_pruned = noisy_y_train[x_mask]

        # Re-weight examples in the loss function for the final fitting
        # s.t. the "apparent" original number of examples in each class
        # is preserved, even though the pruned sets may differ.
        self.sample_weight = np.ones(np.shape(s_pruned))
        for k in range(self.K):
            sample_weight_k = 1.0 / self.noise_matrix[k][k]
            self.sample_weight[s_pruned == k] = sample_weight_k

        train_loader = self._make_dataloader(
            input_info_labels_to_tensordataset(x_pruned, self.sample_weight, s_pruned)
        )

        self._train_loop(train_loader, use_sample_weights=True, print_progress=False)
