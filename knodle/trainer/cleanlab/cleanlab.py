import logging

import numpy as np
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.util import value_counts
from skorch import NeuralNetClassifier
from torch.utils.data import TensorDataset


from knodle.trainer import MajorityVoteTrainer
from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig
from knodle.trainer.cleanlab.pruning import get_noise_indices
from knodle.trainer.cleanlab.psx_estimation import calculate_psx
from knodle.trainer.cleanlab.noisy_matrix_estimation import calculate_noise_matrix
from knodle.trainer.cleanlab.utils import calculate_sample_weights
from knodle.transformation.majority import input_to_majority_vote_input
from knodle.transformation.torch_input import dataset_to_numpy_input

logger = logging.getLogger(__name__)


@AutoTrainer.register('cleanlab')
class CleanLabTrainer(MajorityVoteTrainer):

    def __init__(self, **kwargs):
        if kwargs.get("trainer_config", None) is None:
            kwargs["trainer_config"] = CleanLabConfig()
        super().__init__(**kwargs)

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ) -> None:

        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)

        if dev_model_input_x is not None and dev_gold_labels_y is not None:
            logger.info("Validation data is not used during Cleanlab training")

        # since CL accepts only sklearn.classifier compliant class -> we wraps the PyTorch model
        self.model = self.wrap_model()

        # CL denoising and training
        rp = LearningWithNoisyLabels(
            clf=self.model,
            seed=self.trainer_config.seed,
            cv_n_folds=self.trainer_config.cv_n_folds,
            prune_method=self.trainer_config.prune_method,
            converge_latent_estimates=self.trainer_config.converge_latent_estimates,
            pulearning=self.trainer_config.pulearning,
            n_jobs=self.trainer_config.n_jobs
        )

        # calculate labels based on t and z; perform additional filtering if applicable
        self.model_input_x, noisy_y_train, self.rule_matches_z = input_to_majority_vote_input(
            self.rule_matches_z,
            self.mapping_rules_labels_t,
            self.model_input_x,
            use_probabilistic_labels=self.trainer_config.use_probabilistic_labels,
            filter_non_labelled=self.trainer_config.filter_non_labelled,
            other_class_id=self.trainer_config.other_class_id
        )

        # turn input to the CL-compatible format
        model_input_x_numpy = dataset_to_numpy_input(self.model_input_x)

        # calculate a psx matrix
        psx = calculate_psx(
            self.model_input_x,
            noisy_y_train,
            self.rule_matches_z,
            self.model,
            self.trainer_config.psx_calculation_method,
            self.trainer_config.output_classes,
            self.trainer_config.cv_n_folds,
            self.trainer_config.seed
        )

        # calculate a noise matrix in advance if applicable
        rp.py, rp.noise_matrix, rp.inv_noise_matrix, rp.confident_joint = calculate_noise_matrix(
            noisy_labels=noisy_y_train,
            psx=psx,
            rule_matches_z=self.rule_matches_z,
            num_classes=self.trainer_config.output_classes,
            noise_matrix=self.trainer_config.noise_matrix,
            calibrate=self.trainer_config.calibrate_cj_matrix
        )

        _ = self.fit(rp, model_input_x_numpy, noisy_y_train, psx)
        logging.info("Training is done.")

    def fit(self, rp: LearningWithNoisyLabels, model_input_x: np.ndarray, noisy_labels: np.ndarray, psx: np.ndarray):

        # Number of classes
        if not self.trainer_config.output_classes:
            rp.K = len(np.unique(noisy_labels))
        else:
            rp.K = self.trainer_config.output_classes

        # 'ps' is p(s=k)
        rp.ps = value_counts(noisy_labels) / float(len(noisy_labels))

        # if pulearning == the integer specifying the class without noise.
        if rp.K == 2 and rp.pulearning is not None:  # pragma: no cover
            # pulearning = 1 (no error in 1 class) implies p(s=1|y=0) = 0
            rp.noise_matrix[rp.pulearning][1 - rp.pulearning] = 0
            rp.noise_matrix[1 - rp.pulearning][1 - rp.pulearning] = 1
            # pulearning = 1 (no error in 1 class) implies p(y=0|s=1) = 0
            rp.inverse_noise_matrix[1 - rp.pulearning][rp.pulearning] = 0
            rp.inverse_noise_matrix[rp.pulearning][rp.pulearning] = 1
            # pulearning = 1 (no error in 1 class) implies p(s=1,y=0) = 0
            rp.confident_joint[rp.pulearning][1 - rp.pulearning] = 0
            rp.confident_joint[1 - rp.pulearning][1 - rp.pulearning] = 1

        rp.noise_mask = get_noise_indices(
            noisy_labels,
            psx,
            confident_joint=rp.confident_joint,
            rule2class=self.mapping_rules_labels_t,
            prune_method=rp.prune_method,
            num_classes=self.trainer_config.output_classes,
            # n_jobs=rp.n_jobs,
            # frac_noise=0.5
        )

        x_mask = ~rp.noise_mask
        model_input_x_pruned = model_input_x[x_mask]
        noisy_labels_pruned = noisy_labels[x_mask]

        rp.sample_weigh = calculate_sample_weights(rp.K, rp.noise_matrix, noisy_labels_pruned)
        rp.clf.fit(model_input_x_pruned, noisy_labels_pruned)

        # Check if sample_weight in clf.fit(). Compatible with Python 2/3.
        # if hasattr(inspect, 'getfullargspec') and 'sample_weight' in inspect.getfullargspec(rp.clf.fit).args:
        #     # Re-weight examples in the loss function for the final fitting
        #     # s.t. the "apparent" original number of examples in each class
        #     # is preserved, even though the pruned sets may differ.
        #     rp.sample_weight = np.ones(np.shape(s_pruned))
        #     for k in range(rp.K):
        #         sample_weight_k = 1.0 / rp.noise_matrix[k][k]
        #         rp.sample_weight[s_pruned == k] = sample_weight_k
        #     rp.clf.fit(x_pruned, s_pruned, sample_weight=rp.sample_weight)
        # else:
        #     rp.clf.fit(x_pruned, s_pruned)      # This is less accurate, but best we can do if no sample_weight.

        return rp.clf

    def wrap_model(self):
        """ The function wraps the PyTorch model to a Sklearn model. """
        return NeuralNetClassifier(
            self.model,
            criterion=self.trainer_config.criterion,
            optimizer=self.trainer_config.optimizer,
            lr=self.trainer_config.lr,
            max_epochs=self.trainer_config.epochs,
            batch_size=self.trainer_config.batch_size,
            train_split=None,
            callbacks="disable",
            device=self.trainer_config.device
        )
