import copy
import logging
from typing import Union, Tuple

import torch
import numpy as np

from torch.utils.data import TensorDataset
from torch.utils.data.dataset import Subset

from knodle.trainer import MajorityVoteTrainer
from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.cleanlab.classification import LearningWithNoisyLabelsTorch
from knodle.trainer.cleanlab.config import CleanLabConfig
from knodle.trainer.cleanlab.pruning import update_t_matrix, update_t_matrix_with_prior
from knodle.trainer.cleanlab.psx_estimation import calculate_psx
from knodle.trainer.cleanlab.noisy_matrix_estimation import calculate_noise_matrix
from knodle.trainer.cleanlab.utils import calculate_threshold, calculate_labels
from knodle.transformation.majority import probabilities_to_majority_vote, z_t_matrices_to_majority_vote_probs
from knodle.transformation.torch_input import input_labels_to_tensordataset

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

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ) -> None:

        end_model = copy.deepcopy(self.model).to(self.trainer_config.device)

        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)
        self.trainer_config.optimizer = self.initialise_optimizer()

        self.model_input_x, self.psx_model_input_x, self.rule_matches_z, noisy_y_train = calculate_labels(
            self.model_input_x, self.psx_model_input_x, self.rule_matches_z, self.mapping_rules_labels_t,
            self.trainer_config
        )

        best_dev_loss = 1000

        for i in range(self.trainer_config.iterations):

            logger.info(f"Iteration: {i}")

            t_matrix_updated = self.denoise_t_matrix(noisy_y_train)

            # todo: write tests that here no denoising is needed (all denoising was already done for BOTH matrices)
            new_noisy_y_train_prob = z_t_matrices_to_majority_vote_probs(self.rule_matches_z, t_matrix_updated)
            new_noisy_y_train = np.apply_along_axis(probabilities_to_majority_vote, axis=1, arr=new_noisy_y_train_prob)

            labels_updated = sum(~np.equal(noisy_y_train, new_noisy_y_train))
            logger.info(f"Labels changed: {labels_updated} out of {len(noisy_y_train)}")
            noisy_y_train = new_noisy_y_train

            train_loader = self._make_dataloader(input_labels_to_tensordataset(self.model_input_x, noisy_y_train))
            self.model = copy.deepcopy(end_model).to(self.trainer_config.device)
            self._train_loop(train_loader, print_progress=False)

            if self.dev_model_input_x:
                clf_report, dev_loss = self.test_with_loss(self.dev_model_input_x, self.dev_gold_labels_y)
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    logger.info(f"Clf_report: {clf_report}, Dev loss: {dev_loss}, denoising continues.")
                else:
                    logger.info(f"The model does not improve on the dev set (previous dev loss: {best_dev_loss}, "
                                f"new dev loss: {dev_loss}). Denoising stops.")
                    break
            else:
                if labels_updated == 0:
                    logger.info("No more iterations since the labels do not change anymore.")
                    break

        logging.info("Training is done.")

    def denoise_t_matrix(self, noisy_y_train):

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

        # calculate threshold values
        thresholds = calculate_threshold(psx, noisy_y_train, self.trainer_config.output_classes)

        # calculate a noise matrix in advance if applicable
        rp.noise_matrix, rp.inv_noise_matrix, rp.confident_joint = calculate_noise_matrix(
            psx, self.rule_matches_z, thresholds,
            num_classes=self.trainer_config.output_classes,
            noise_matrix=self.trainer_config.noise_matrix,
            calibrate=self.trainer_config.calibrate_cj_matrix
        )

        if self.trainer_config.use_prior:
            # 2nd method: normalized((t matrix * prior) + confident_joint)
            t_matrix_updated = update_t_matrix_with_prior(rp.confident_joint, self.mapping_rules_labels_t)
        else:
            # 1st method: p * (normalized confident_joint) + p * (t matrix)
            t_matrix_updated = update_t_matrix(rp.confident_joint, self.mapping_rules_labels_t, p=self.trainer_config.p)

        logger.info(t_matrix_updated)
        return t_matrix_updated

    def get_pruned_input(self, x_mask, noisy_labels) -> Tuple[Union[np.ndarray, TensorDataset], np.ndarray, np.ndarray]:

        noisy_labels_pruned = noisy_labels[x_mask]
        rule_matches_z_pruned = self.rule_matches_z[x_mask]

        if isinstance(self.model_input_x, np.ndarray):
            return self.model_input_x[x_mask], noisy_labels_pruned, rule_matches_z_pruned

        elif isinstance(self.model_input_x, TensorDataset):
            # todo: write tests
            model_input_x_pruned_subset = [data for data in Subset(self.model_input_x, np.where(x_mask)[0])]
            model_input_x_pruned = TensorDataset(*[
                torch.stack(([tensor[i] for tensor in model_input_x_pruned_subset]))
                for i in range(len(self.model_input_x.tensors))
            ])
            return model_input_x_pruned, noisy_labels_pruned, rule_matches_z_pruned

        else:
            raise ValueError("Unknown input format")

    # def fit(self, rp: LearningWithNoisyLabelsTorch, noisy_labels: np.ndarray, psx: np.ndarray):
    #
    #     # todo: add rp.pulearning not None
    #
    #     rp.ps = value_counts(noisy_labels) / float(len(noisy_labels))       # 'ps' is p(s=k)
    #
    #     model_input_x_pruned, noisy_labels_pruned, rule_matches_z_pruned = self.get_pruned_input(
    #         ~label_errors_mask, noisy_labels
    #     )
    #     # todo: add sample weights (currently: only the function from the original CL)
    #     rp.sample_weight = calculate_sample_weights(
    #         rp.K, label_errors_mask, noisy_labels_pruned, self.mapping_rules_labels_t, rule_matches_z_pruned
    #     )
    #     train_loader = self._make_dataloader(
    #         input_labels_to_tensordataset(model_input_x_pruned, noisy_labels_pruned)
    #     )




'''
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
