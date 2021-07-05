import logging

import numpy as np
from cleanlab.classification import LearningWithNoisyLabels
from skorch import NeuralNetClassifier
from torch.utils.data import TensorDataset

from knodle.trainer import MajorityVoteTrainer
from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig
from knodle.trainer.cleanlab.latent_estimation import estimate_cv_predicted_probabilities_split_by_rules, \
    estimate_cv_predicted_probabilities_split_by_signatures
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
        self._apply_rule_reduction()

        if dev_model_input_x is not None and dev_gold_labels_y is not None:
            logger.info("Validation data is not used during Cleanlab training")

        # since CL accepts only sklearn.classifier compliant class, we wraps the PyTorch model
        self.model = NeuralNetClassifier(
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

        # calculate labels based on t and z; perform additional filtering if applicable
        self.model_input_x, noisy_y_train, self.rule_matches_z = input_to_majority_vote_input(
            self.rule_matches_z, self.mapping_rules_labels_t, self.model_input_x,
            use_probabilistic_labels=self.trainer_config.use_probabilistic_labels,
            filter_non_labelled=self.trainer_config.filter_non_labelled,
            other_class_id=self.trainer_config.other_class_id,)

        # turn input to the CL-compatible format
        model_input_x_numpy = dataset_to_numpy_input(self.model_input_x)

        # calculate psx in advance with splitting by rules
        if self.trainer_config.psx_calculation_method == "rules":
            psx = estimate_cv_predicted_probabilities_split_by_rules(
                self.model_input_x, noisy_y_train, self.rule_matches_z, self.model, self.trainer_config.output_classes,
                seed=self.trainer_config.seed, cv_n_folds=self.trainer_config.cv_n_folds
            )

        elif self.trainer_config.psx_calculation_method == "signatures":
            psx = estimate_cv_predicted_probabilities_split_by_signatures(
                self.model_input_x, noisy_y_train, self.rule_matches_z, self.model, self.trainer_config.output_classes,
                seed=self.trainer_config.seed, cv_n_folds=self.trainer_config.cv_n_folds
            )

        elif self.trainer_config.psx_calculation_method == "random":
            # if no psx calculation method is specified, psx will be calculated in CL with random folder splitting
            psx = None

        else:
            raise ValueError("Unknown psx calculation method.")

        # CL denoising and training
        rp = LearningWithNoisyLabels(
            clf=self.model, seed=self.trainer_config.seed,
            cv_n_folds=self.trainer_config.cv_n_folds,
            prune_method=self.trainer_config.prune_method,
            converge_latent_estimates=self.trainer_config.converge_latent_estimates,
            pulearning=self.trainer_config.pulearning,
            n_jobs=self.trainer_config.n_jobs
        )
        _ = rp.fit(model_input_x_numpy, noisy_y_train, psx=psx)
        logging.info("Training is done.")

