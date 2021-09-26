import logging

import torch
import numpy as np
from cleanlab.classification import LearningWithNoisyLabels
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import Subset

from knodle.trainer import MajorityVoteTrainer
from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.cleanlab.classification import LearningWithNoisyLabelsTorch
from knodle.trainer.cleanlab.config import CleanLabConfig
from knodle.trainer.cleanlab.pruning import get_noise_indices
from knodle.trainer.cleanlab.psx_estimation import calculate_psx
from knodle.trainer.cleanlab.noisy_matrix_estimation import calculate_noise_matrix
from knodle.trainer.cleanlab.utils import calculate_sample_weights
from knodle.transformation.majority import input_to_majority_vote_input, probabilities_to_majority_vote
from knodle.transformation.torch_input import dataset_to_numpy_input, input_labels_to_tensordataset

logger = logging.getLogger(__name__)


@AutoTrainer.register('cleanlab')
class CleanLabTrainer(MajorityVoteTrainer):

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

            # todo: check the compatibility of criterion for correct loss calculation

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ) -> None:

        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)

        if dev_model_input_x is not None and dev_gold_labels_y is not None:
            logger.info("Validation data is not used in Cleanlab training")

        # CL accepts only sklearn.classifier compliant class -> we wrap the PyTorch models with Scorch
        # self.psx_model = wrap_model_default(self.psx_model, self.trainer_config)

        # todo: current testing: noisy_y_train = matrix with probs (samples x classes), NOT ONE HOT - test differently
        #  as well

        # calculate labels based on t and z; perform additional filtering if applicable
        self.model_input_x, noisy_y_train, self.rule_matches_z = input_to_majority_vote_input(
            self.rule_matches_z,
            self.mapping_rules_labels_t,
            self.model_input_x,
            use_probabilistic_labels=self.trainer_config.use_probabilistic_labels,
            filter_non_labelled=self.trainer_config.filter_non_labelled,
            other_class_id=self.trainer_config.other_class_id
        )

        # train simple baseline with the original CL
        if self.trainer_config.train_baseline:
            model_input_x_numpy = dataset_to_numpy_input(self.model_input_x)
            return self.train_baseline(model_input_x_numpy, noisy_y_train)

        # CL denoising and training
        rp = LearningWithNoisyLabelsTorch(
            clf=self.model,
            seed=self.trainer_config.seed,
            cv_n_folds=self.trainer_config.cv_n_folds,
            prune_method=self.trainer_config.prune_method,
            converge_latent_estimates=self.trainer_config.converge_latent_estimates,
            pulearning=self.trainer_config.pulearning,
            n_jobs=self.trainer_config.n_jobs
        )

        # calculate a psx matrix
        psx = calculate_psx(
            self.psx_model_input_x,
            noisy_y_train,
            self.rule_matches_z,
            self.psx_model,
            config=self.trainer_config
        )

        # calculate thresholds per class
        # P(we predict the given noisy label is k | given noisy label is k) - the same way it is done in the original CL
        if self.trainer_config.use_probabilistic_labels:
            noisy_y_train_labels = np.apply_along_axis(probabilities_to_majority_vote, axis=1, arr=noisy_y_train)
            thresholds = np.asarray(
                [np.mean(psx[:, k][np.asarray(noisy_y_train_labels) == k]) for k in
                 range(self.trainer_config.output_classes)]
            )
        else:
            thresholds = np.asarray(
                [np.mean(psx[:, k][np.asarray(noisy_y_train) == k]) for k in
                 range(self.trainer_config.output_classes)]
            )

        # calculate a noise matrix in advance if applicable
        rp.noise_matrix, rp.inv_noise_matrix, rp.confident_joint = calculate_noise_matrix(
            noisy_y_train,
            psx=psx,
            rule_matches_z=self.rule_matches_z,
            thresholds=thresholds,
            num_classes=self.trainer_config.output_classes,
            noise_matrix=self.trainer_config.noise_matrix,
            calibrate=self.trainer_config.calibrate_cj_matrix
        )

        _ = self.fit(rp, self.model_input_x, noisy_y_train, psx)
        logging.info("Training is done.")

    def fit(self, rp: LearningWithNoisyLabelsTorch, model_input_x: TensorDataset, noisy_labels: np.ndarray, psx: np.ndarray):

        # todo: add rp.pulearning not None
        # Number of classes
        if not self.trainer_config.output_classes:
            rp.K = len(np.unique(noisy_labels))
        else:
            rp.K = self.trainer_config.output_classes

        # rp.ps = value_counts(noisy_labels) / float(len(noisy_labels))       # 'ps' is p(s=k)
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

        if isinstance(model_input_x, np.ndarray):
            model_input_x_pruned = model_input_x[x_mask]

        elif isinstance(model_input_x, TensorDataset):
            # todo: write tests
            model_input_x_pruned_subset = [data for data in Subset(model_input_x, np.where(x_mask)[0])]
            model_input_x_pruned = TensorDataset(*[
                torch.stack(([tensor[i] for tensor in model_input_x_pruned_subset]))
                for i in range(len(model_input_x.tensors))
            ])

        else:
            raise ValueError("Unknown input format")

        noisy_labels_pruned = noisy_labels[x_mask]
        rule_matches_z_pruned = self.rule_matches_z[x_mask]

        rp.sample_weight = calculate_sample_weights(
            rp.K, rp.noise_matrix, noisy_labels_pruned, self.mapping_rules_labels_t, rule_matches_z_pruned
        )

        train_loader = self._make_dataloader(
            input_labels_to_tensordataset(model_input_x_pruned, noisy_labels_pruned)
        )
        self.trainer_config.optimizer = self.initialise_optimizer()
        self._train_loop(train_loader)

    def train_baseline(self, model_input_x: np.ndarray, noisy_labels: np.ndarray):
        rp = LearningWithNoisyLabels(clf=self.model, seed=self.trainer_config.seed)
        _ = rp.fit(model_input_x, noisy_labels)
        return rp.clf




'''
Ben: 
1) calculate a threshold per class (the average probability of a class for instances labeled as the class)
        -> average probability that we expect for this class 
2) C matrix (LFs x classes): how many instances for one LF are confidently labeled as the class of the LF
3) cross-validation: let the other LFs train the classifier -> see, how many examples of the left out LFs are above the 
    average for this relation or not 
        - for each instance with LF: does the probability of any class C* exceed the threshold? 
        if yes, increase the count of row of the LF. 
    
In other words: 
1) thresholds: as in the original paper (per class)
2) instead of noisy labels, take LFs 
    - for each sample, where e.g. LF1 matched:
        - look at the predicted class for this sample 
        - if the prob exceeds the threshold for this class: count 
        
pruning: pro LF
'''