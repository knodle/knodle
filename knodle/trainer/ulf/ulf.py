import copy
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import Module
from torch.utils.data import TensorDataset

from knodle.trainer import MajorityVoteTrainer
from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.ulf.config import UlfConfig
from knodle.trainer.ulf.utils import update_t_matrix, update_t_matrix_with_prior
from knodle.trainer.ulf.noisy_matrix_estimation import calculate_noise_matrix
from knodle.trainer.cleanlab.psx_estimation import calculate_psx
from knodle.trainer.cleanlab.utils import calculate_threshold
from knodle.transformation.majority import input_to_majority_vote_input, z_t_matrices_to_labels
from knodle.transformation.torch_input import input_labels_to_tensordataset

logger = logging.getLogger(__name__)


@AutoTrainer.register('ulf')
class UlfTrainer(MajorityVoteTrainer):

    def __init__(
            self,
            psx_model: Module = None,
            psx_model_input_x: np.ndarray = None,
            df_train: pd.DataFrame = None,          # todo: to delete
            output_file: str = None,          # todo: to delete
            **kwargs):

        self.psx_model = psx_model if psx_model else kwargs.get("model")
        self.psx_model_input_x = psx_model_input_x if psx_model_input_x else kwargs.get("model_input_x")

        self.df_train = df_train        # todo: to delete
        self.output_file = output_file        # todo: to delete

        if kwargs.get("trainer_config", None) is None:
            kwargs["trainer_config"] = UlfConfig()
        super().__init__(**kwargs)

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ) -> None:
        max_patience = 3
        best_dev_loss = 1000
        patience = 0

        # save the original non-trained model in order to copy it for further trainings
        end_model = copy.deepcopy(self.model).to(self.trainer_config.device)
        opt = copy.deepcopy(self.trainer_config.optimizer)

        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)

        empty_rules = np.argwhere(np.all(self.rule_matches_z[..., :] == 0, axis=0))
        self.rule_matches_z = np.delete(self.rule_matches_z, empty_rules, axis=1)
        self.mapping_rules_labels_t = np.delete(self.mapping_rules_labels_t, empty_rules, axis=0)

        samples_without_matches = np.argwhere(np.all(self.rule_matches_z[..., :] == 0, axis=1)).squeeze().tolist()

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

        for i in range(self.trainer_config.iterations):

            logger.info(f"Iteration: {i + 1}")

            # update the t matrix and recalculate the labels
            t_matrix_updated, labels_to_subst = self.denoise_t_matrix(noisy_y_train, samples_without_matches)
            noisy_y_train, num_labels_upd, idx_upd = self.get_updated_labels(t_matrix_updated, noisy_y_train, labels_to_subst)

            # updated_samples = pd.DataFrame(
            #     {
            #         "sample": self.df_train["sample"].iloc[idx_upd],
            #         "old_label": noisy_y_train[idx_upd],
            #         "upd_label": noisy_y_train_upd[idx_upd]
            #     }
            # )
            # updated_samples.to_csv(f"{self.output_file}_{i}.csv", index=None)

            # create the dataset
            train_loader = self._make_dataloader(input_labels_to_tensordataset(self.model_input_x, noisy_y_train))

            # initialize the optimizer and the model anew
            self.model = copy.deepcopy(end_model).to(self.trainer_config.device)
            self.trainer_config.optimizer = self.initialise_optimizer(opt)

            # train the model
            self._train_loop(train_loader)

            # early stopping: either the labels do not change anymore or the dev loss does not improve
            if not self.dev_model_input_x:
                if num_labels_upd == 0:
                    logger.info("No more iterations since the labels do not change anymore.")
                    break
            else:
                clf_report, dev_loss = self.test_with_loss(self.dev_model_input_x, self.dev_gold_labels_y)
                logger.info(f"Clf_report: {clf_report}")

                if dev_loss < best_dev_loss:
                    patience = 0
                    best_dev_loss = dev_loss
                    logger.info(f"Dev loss: {dev_loss}, denoising continues.")
                    os.makedirs(self.trainer_config.save_model_path, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(
                        self.trainer_config.save_model_path, f"{self.trainer_config.save_model_name}_fin.pt"
                    ))
                else:
                    patience += 1
                    if patience == max_patience:
                        logger.info(f"The model does not improve on the dev set (previous dev loss: {best_dev_loss}, "
                                    f"new dev loss: {dev_loss}). Denoising stops.")
                        break
                    else:
                        logger.info(f"The model does not improve on the dev set (previous dev loss: {best_dev_loss}, "
                                    f"new dev loss: {dev_loss}). Patience now equals {patience}")

        logging.info(
            f"All iterations are completed, {i+1} iterations performed. The trained model will be tested on the "
            f"test set now..."
        )

        if self.trainer_config.early_stopping:
            self.load_model(f"{self.trainer_config.save_model_name}_fin.pt", self.trainer_config.save_model_path)
            logger.info("The best model on dev set will be used for evaluation. ")

    def denoise_t_matrix(self, noisy_y_train: np.ndarray, samples_without_matches) -> np.ndarray:
        """
        Recalculation of labeling functions to labels matrix
        :param noisy_y_train: original t (labeling functions x labels) matrix
        :return: updated labeling functions x labels matrix
        """
        # calculate a psx matrix
        psx = calculate_psx(
            self.psx_model_input_x,
            noisy_y_train,
            self.rule_matches_z,
            self.psx_model,
            config=self.trainer_config
        )
        labels_to_subst = {}
        for idx in samples_without_matches:
            labels_to_subst[idx] = int(np.argmax(psx[idx, :]))

        # calculate threshold values
        thresholds = calculate_threshold(psx, noisy_y_train, self.trainer_config.output_classes)

        # calculate a noise matrix in advance if applicable
        noise_matrix, inv_noise_matrix, confident_joint = calculate_noise_matrix(
            psx, self.rule_matches_z, thresholds,
            num_classes=self.trainer_config.output_classes,
            noise_matrix=self.trainer_config.noise_matrix,
            calibrate=self.trainer_config.calibrate_cj_matrix
        )

        if self.trainer_config.use_prior:
            # normalized((t matrix * prior) + confident_joint)
            return update_t_matrix_with_prior(
                confident_joint, self.mapping_rules_labels_t, verbose=self.trainer_config.verbose
            ), labels_to_subst
        else:
            # p * (normalized confident_joint) + (1 - p) * (t matrix)
            return update_t_matrix(
                confident_joint, self.mapping_rules_labels_t, self.trainer_config.p, verbose=self.trainer_config.verbose
            ), labels_to_subst

    def get_updated_labels(
            self, t_matrix_updated: np.ndarray, noisy_y_train: np.ndarray, labels_to_subst
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        This function recalculates the labels based on the updated t matrix.
        :param t_matrix_updated:
        :param noisy_y_train:
        :return:
        """
        # todo: check whether any filtering happened (actually shouldn't....)
        new_noisy_y_train = z_t_matrices_to_labels(self.rule_matches_z, t_matrix_updated)

        for id, label in labels_to_subst.items():
            new_noisy_y_train[id] = label

        num_labels_updated = sum(~np.equal(noisy_y_train, new_noisy_y_train))
        updated_samples_idx = np.where(~np.equal(noisy_y_train, new_noisy_y_train))[0]
        logger.info(f"Labels changed: {num_labels_updated} out of {len(noisy_y_train)}")
        return new_noisy_y_train, num_labels_updated, updated_samples_idx

    def initialise_optimizer(self, custom_opt):
        try:
            return custom_opt(params=self.model.parameters(), lr=self.trainer_config.lr)
        except TypeError:
            logger.info(
                "Wrong optimizer parameters. Optimizer should belong to torch.optim class or be PyTorch compatible."
            )
