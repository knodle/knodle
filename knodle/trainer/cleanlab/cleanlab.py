import logging

import numpy as np
from cleanlab.pruning import get_noise_indices
from cleanlab.util import value_counts
from torch.utils.data import TensorDataset

from knodle.trainer import MajorityVoteTrainer
from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig
from knodle.trainer.cleanlab.psx_estimation import estimate_py_noise_matrices_and_cv_pred_proba
from knodle.transformation.majority import input_to_majority_vote_input
from knodle.transformation.torch_input import input_info_labels_to_tensordataset

logger = logging.getLogger(__name__)


@AutoTrainer.register('cleanlab')
class CleanlabTrainer(MajorityVoteTrainer):

    def __init__(
            self,
            psx_model=None,
            psx_model_input_x=None,
            **kwargs
    ):

        self.psx_model = psx_model if psx_model else kwargs.get("model")
        self.psx_model_input_x = psx_model_input_x if psx_model_input_x else kwargs.get("model_input_x")

        if kwargs.get("trainer_config", None) is None:
            kwargs["trainer_config"] = CleanLabConfig()
        super().__init__(**kwargs)

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ) -> None:

        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)
        self.trainer_config.optimizer = self.initialise_optimizer()

        self.model_input_x, self.psx_model_input_x, noisy_y_train, self.rule_matches_z = input_to_majority_vote_input(
            self.rule_matches_z,
            self.mapping_rules_labels_t,
            self.model_input_x,
            self.psx_model_input_x,
            use_probabilistic_labels=self.trainer_config.use_probabilistic_labels,
            filter_non_labelled=self.trainer_config.filter_non_labelled,
            probability_threshold=self.trainer_config.probability_threshold,
            other_class_id=self.trainer_config.other_class_id,
            choose_random_label=self.trainer_config.choose_random_label
        )

        # 'ps' is p(s=k)
        ps = value_counts(noisy_y_train) / float(len(noisy_y_train))

        py, noise_matrix, inverse_noise_matrix, confident_joint, psx = \
            estimate_py_noise_matrices_and_cv_pred_proba(
                data_features=self.psx_model_input_x,
                labels=noisy_y_train,
                model=self.psx_model,
                config=self.trainer_config,
                thresholds=None
            )

        # if pulearning == the integer specifying the class without noise.
        if self.trainer_config.output_classes == 2 and self.trainer_config.pulearning is not None:  # pragma: no cover
            # pulearning = 1 (no error in 1 class) implies p(s=1|y=0) = 0
            noise_matrix[self.trainer_config.pulearning][1 - self.trainer_config.pulearning] = 0
            noise_matrix[1 - self.trainer_config.pulearning][1 - self.trainer_config.pulearning] = 1
            # pulearning = 1 (no error in 1 class) implies p(y=0|s=1) = 0
            inverse_noise_matrix[1 - self.trainer_config.pulearning][self.trainer_config.pulearning] = 0
            inverse_noise_matrix[self.trainer_config.pulearning][self.trainer_config.pulearning] = 1
            # pulearning = 1 (no error in 1 class) implies p(s=1,y=0) = 0
            confident_joint[self.trainer_config.pulearning][1 - self.trainer_config.pulearning] = 0
            confident_joint[1 - self.trainer_config.pulearning][1 - self.trainer_config.pulearning] = 1

        # This is the actual work of this function.

        # Get the indices of the examples we wish to prune
        noise_mask = get_noise_indices(
            noisy_y_train,
            psx,
            inverse_noise_matrix=inverse_noise_matrix,
            confident_joint=confident_joint,
            prune_method=self.trainer_config.prune_method,
            n_jobs=self.trainer_config.n_jobs,
        )

        x_mask = ~noise_mask
        s_pruned = noisy_y_train[x_mask]
        x_pruned = TensorDataset(*[inp[x_mask] for inp in self.model_input_x.tensors])

        # Re-weight examples in the loss function for the final fitting s.t. the "apparent" original number of examples
        # in each class is preserved, even though the pruned sets may differ.
        sample_weight = np.ones(np.shape(s_pruned))
        for k in range(self.trainer_config.output_classes):
            sample_weight_k = 1.0 / noise_matrix[k][k]
            sample_weight[s_pruned == k] = sample_weight_k

        train_loader = self._make_dataloader(
            input_info_labels_to_tensordataset(x_pruned, sample_weight, s_pruned)
        )

        self._train_loop(train_loader, use_sample_weights=True)
