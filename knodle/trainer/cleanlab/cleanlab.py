import copy
import logging

import numpy as np
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.util import value_counts
from skorch import NeuralNetClassifier
from torch.utils.data import TensorDataset


from knodle.trainer import MajorityVoteTrainer
from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig
from knodle.trainer.cleanlab.model_wrapper import wrap_model
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

        self.pytorch_model = copy.deepcopy(self.model).to(self.trainer_config.device)

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ) -> None:

        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)

        if dev_model_input_x is not None and dev_gold_labels_y is not None:
            logger.info("Validation data is not used during Cleanlab training")

        # since CL accepts only sklearn.classifier compliant class -> we wraps the PyTorch model
        self.model = NeuralNetClassifier(
            self.model,
            criterion=self.trainer_config.criterion,
            # optimizer=self.trainer_config.optimizer,
            # lr=self.trainer_config.lr,
            max_epochs=self.trainer_config.epochs,
            batch_size=self.trainer_config.batch_size,
            train_split=None,
            callbacks="disable",
            device=self.trainer_config.device
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

        if self.trainer_config.train_baseline:
            return self.train_baseline(model_input_x_numpy, noisy_y_train)

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

        # calculate a psx matrix
        psx = calculate_psx(
            model_input_x=self.model_input_x,
            noisy_labels=noisy_y_train,
            rule_matches_z=self.rule_matches_z,
            model=self.model,
            psx_calculation_method=self.trainer_config.psx_calculation_method,
            num_classes=self.trainer_config.output_classes,
            cv_n_folds=self.trainer_config.cv_n_folds,
            seed=self.trainer_config.seed
        )

        # calculate thresholds per class
        # P(we predict the given noisy label is k | given noisy label is k) - the same way it is done in the original CL
        thresholds = np.asarray(
            [np.mean(psx[:, k][np.asarray(noisy_y_train) == k]) for k in range(self.trainer_config.output_classes)]
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

        _ = self.fit(rp, model_input_x_numpy, noisy_y_train, psx)
        logging.info("Training is done.")

    def fit(self, rp: LearningWithNoisyLabels, model_input_x: np.ndarray, noisy_labels: np.ndarray, psx: np.ndarray):

        self.model = wrap_model(self.pytorch_model, self.trainer_config)

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
        model_input_x_pruned = model_input_x[x_mask]
        noisy_labels_pruned = noisy_labels[x_mask]
        rule_matches_z_pruned = self.rule_matches_z[x_mask]

        rp.sample_weight = calculate_sample_weights(
            rp.K, rp.noise_matrix, noisy_labels_pruned, self.mapping_rules_labels_t, rule_matches_z_pruned
        )

        # in order to train a skorch model with sample weights, we need to pass all the data as dicts - #todo tests lack
        model_input_x_pruned = {'X': model_input_x_pruned, 'sample_weight': rp.sample_weight}
        self.model.fit(model_input_x_pruned, noisy_labels_pruned)

        return self.model

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