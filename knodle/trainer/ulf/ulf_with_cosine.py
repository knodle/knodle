import copy
import logging
import os
from typing import Tuple, Dict

import numpy as np
import torch
from snorkel.utils import probs_to_preds
from torch.nn import Module
from torch.utils.data import TensorDataset

from examples.trainer.ulf.utils import WrenchDataset
from knodle.trainer import MajorityVoteTrainer
from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.cleanlab.utils import calculate_threshold
from knodle.trainer.ulf.config import UlfConfig
from knodle.trainer.ulf.noisy_matrix_estimation import calculate_noise_matrix
from knodle.trainer.ulf.utils import update_t_matrix, update_t_matrix_with_prior
from knodle.trainer.utils import log_section
from knodle.trainer.utils.split import k_folds_splitting_by_signatures
from knodle.transformation.majority import z_t_matrices_to_labels, z_t_matrices_to_majority_vote_probs, \
    probabilities_to_majority_vote
from wrench.evaluation import METRIC

logger = logging.getLogger(__name__)


@AutoTrainer.register('ulf')
class UlfTrainer(MajorityVoteTrainer):

    def __init__(
            self,
            psx_model: Module = None,
            psx_model_input_x: TensorDataset = None,
            **kwargs):

        self.psx_model = psx_model if psx_model else kwargs.get("model")
        self.psx_model_input_x = psx_model_input_x if psx_model_input_x else kwargs.get("model_input_x")

        if kwargs.get("trainer_config", None) is None:
            kwargs["trainer_config"] = UlfConfig()
        super().__init__(**kwargs)

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None,
            test_data=None, target=None
    ):
        results = []
        best_metric_w_w, best_metric_fs_w, best_metric_w_fs, best_metric_fs_fs = 0, 0, 0, 0
        best_set_w_w, best_set_fs_w, best_set_w_fs, best_set_fs_fs = "", "", "", ""
        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)

        noisy_y_train = z_t_matrices_to_majority_vote_probs(self.rule_matches_z, self.mapping_rules_labels_t)
        # noisy_y_train = np.apply_along_axis(probabilities_to_majority_vote, axis=1, arr=noisy_y_train)

        noisy_y_train = probs_to_preds(noisy_y_train)
        noisy_y_train_fs = noisy_y_train
        noisy_y_train_w = noisy_y_train

        start_model = copy.deepcopy(self.model)

        for i in range(self.trainer_config.iterations):
            logger.info(f"Iteration: {i + 1}")
            iter_results = {}

            # recalculate the labels
            updated_t_matrix_fs, updated_t_matrix_w = self.denoise_t_matrix(noisy_y_train_fs, noisy_y_train_w, start_model)  # labels_to_subst

            # update the t matrix after first stage CV
            log_section("CV was done with first stage of COSINE only", logger)
            self.mapping_rules_labels_t = updated_t_matrix_fs
            print(self.mapping_rules_labels_t)
            metrics, y_hard_fs, best_metric_fs_fs, best_metric_fs_w, best_set_fs_fs, best_set_fs_w \
                = self.train_end_model(noisy_y_train_fs, start_model, test_data, i, best_metric_fs_fs, best_metric_fs_w,
                                       best_set_fs_fs, best_set_fs_w, target)
            iter_results["CV_fs"] = metrics

            # update the t matrix after whole cosine CV
            log_section("CV was done with the whole COSINE", logger)
            self.mapping_rules_labels_t = updated_t_matrix_w
            print(self.mapping_rules_labels_t)
            metrics, y_hard_w, best_metric_w_fs, best_metric_w_w, best_set_w_fs, best_set_w_w \
                = self.train_end_model(noisy_y_train_w, start_model, test_data, i, best_metric_w_fs, best_metric_w_w,
                                       best_set_w_fs, best_set_w_w, target)
            iter_results["CV_whole"] = metrics

            # update noisy labels
            noisy_y_train_fs = y_hard_fs
            noisy_y_train_w = y_hard_w

            results.append(iter_results)

        self.print_results(results)
        print(f"CV with first stage Cosine, end training with first stage Cosine. Best value = {best_metric_fs_fs}. {best_set_fs_fs}")
        print(f"CV with whole Cosine, end training with first stage Cosine. Best value = {best_metric_w_fs}. {best_set_w_fs}")
        print(f"CV with first stage Cosine, end training with whole Cosine. Best value = {best_metric_fs_w}. {best_set_fs_w}")
        print(f"CV with whole Cosine, end training with whole Cosine. Best value = {best_metric_w_w}. {best_set_w_w}")

        return results

    def train_end_model(
            self, noisy_y_train, start_model, test_data, i, best_metric_fs, best_metric_w, best_set_fs, best_set_w, target
    ):

        # recalculate noisy labels
        y_hard, y_soft, num_labels_upd, idx_upd = self.get_updated_labels(noisy_y_train)

        # try end cosine model
        metrics = {"fs": {}, "whole": {}}

        ##############
        ##############

        logger.info("END MODEL: Train full Cosine model with hard labels...")
        self.model = copy.deepcopy(start_model)
        hard_metrics = self.train_n_get_metrics_w(test_data, y_hard, i, if_soft=False)
        metrics["whole"]["hard"] = hard_metrics
        # if there is a better model, save it
        best_metric_w, if_best = self.check_metric_value(hard_metrics, best_metric_w, target,
                                                         f"{self.trainer_config.save_model_name}_w_cosine")
        best_set_w = f"Best setting: Iteration_{i}_hard_label" if if_best else best_set_w

        ##############
        ##############

        logger.info("END MODEL: Train full Cosine mode with soft labels...")
        self.model = copy.deepcopy(start_model)
        soft_metrics = self.train_n_get_metrics_w(test_data, y_soft, i, if_soft=True)
        metrics["whole"]["soft"] = soft_metrics
        # if there is a better model, save it
        best_metric_w, if_best = self.check_metric_value(soft_metrics, best_metric_w, target,
                                                         f"{self.trainer_config.save_model_name}_w_cosine")
        best_set_w = f"Best setting: Iteration_{i}_soft_label" if if_best else best_set_w

        ##############
        ##############

        logger.info("END MODEL: Train only first stage of Cosine model (= pretraining) with hard labels...")
        self.model = copy.deepcopy(start_model)
        hard_metrics = self.train_n_get_metrics_pretraining(test_data, y_hard, i, if_soft=False)
        metrics["fs"]["hard"] = hard_metrics
        # if there is a better model, save it
        best_metric_fs, if_best = self.check_metric_value(hard_metrics, best_metric_fs, target, f"{self.trainer_config.save_model_name}_fs_cosine")
        best_set_fs = f"Best setting: Iteration_{i}_hard_label" if if_best else best_set_fs

        ##############
        ##############

        logger.info("END MODEL: Train only first stage of Cosine model (= pretraining) with soft labels...")
        self.model = copy.deepcopy(start_model)
        soft_metrics = self.train_n_get_metrics_pretraining(test_data, y_soft, i, if_soft=True)
        metrics["fs"]["soft"] = soft_metrics
        # if there is a better model, save it
        best_metric_fs, if_best = self.check_metric_value(soft_metrics, best_metric_fs, target,
                                                          f"{self.trainer_config.save_model_name}_fs_cosine")
        best_set_fs = f"Best setting: Iteration_{i}_hard_label" if if_best else best_set_fs

        ##############
        ##############

        logger.info(f"Iteration {i}, results: {str(metrics)}")

        return metrics, y_hard, best_metric_fs, best_metric_w, best_set_fs, best_set_w

    def check_metric_value(self, metrics, best, target, model_name):
        try:
            if metrics[target] > best:
                self.model.save(os.path.join(self.trainer_config.save_model_path, model_name))
                print(
                    f"New best metric: {metrics[target]}, new model is saved to {os.path.join(self.trainer_config.save_model_path, model_name)}")
                return metrics[target], True
            else:
                print(f"The current metric {metrics[target]} is worse than the best one {best}; no model is saved.")
                return best, False
        except KeyError:
            print("Please! Check target metric!")

    def print_results(self, results):
        log_section(f"Intermediate results: ", logger)

        for n_iter, iter_result in enumerate(results):
            log_section(f"Iteration {n_iter}", logger)
            logging.info(f"Iteration {n_iter}, CV_fs results: {str(iter_result['CV_fs'])}, CV_whole results: {str(iter_result['CV_whole'])}")

        log_section(
            f"All iterations are completed, {self.trainer_config.iterations} iterations performed. "
            f"The trained model will be tested on the test set now...", logger
        )

    def train_n_get_metrics_w(self, test_data, noisy_y_train, i, if_soft):
        metrics = {}
        # try full cosine
        print("Fitting the whole cosine....")
        self.model.fit(
            dataset_train=self.model_input_x, dataset_valid=self.dev_model_input_x, y_train=noisy_y_train,
            soft_labels=if_soft, device=self.trainer_config.device, metric=self.trainer_config.target,
            patience=self.trainer_config.patience, evaluation_step=self.trainer_config.evaluation_step,
            seed=self.trainer_config.seed
        )
        print(
            f"if_soft = {if_soft}. Full Cosine model. Iteration {i + 1} out of {self.trainer_config.iterations} is done.")
        for metric in METRIC:
            curr_metric = self.model.test(test_data, metric)
            print(f"{metric} is {curr_metric}")
            metrics[metric] = curr_metric
        return metrics

    def train_n_get_metrics_pretraining(self, test_data, noisy_y_train, i, if_soft):
        metrics = {}
        print("Fitting the first part of cosine (pretraining)....")
        self.model.fit_fs(
            dataset_train=self.model_input_x, dataset_valid=self.dev_model_input_x, y_train=noisy_y_train,
            soft_labels=if_soft, device=self.trainer_config.device, metric=self.trainer_config.target,
            patience=self.trainer_config.patience, evaluation_step=self.trainer_config.evaluation_step,
            seed=self.trainer_config.seed
        )
        print(
            f"if_soft = {if_soft}. First stage of Cosine model. Iteration {i + 1} out of {self.trainer_config.iterations} is done.")
        for metric in METRIC:
            curr_metric = self.model.test(test_data, metric)
            print(f"{metric} is {curr_metric}")
            metrics[metric] = curr_metric
        return metrics

    def denoise_t_matrix(self, noisy_y_train_fs: np.ndarray, noisy_y_train_w: np.ndarray, start_model) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recalculation of labeling functions to labels matrix
        :param noisy_y_train: original t (labeling functions x labels) matrix
        :param start_model: raw untrained model which will be copied in each cross-validation batch training
        :return: updated labeling functions x labels matrix
        """

        non_labeled_samples = list(np.where(np.all(self.rule_matches_z == 0, axis=1))[0])

        if self.trainer_config.psx_calculation_method == "signatures":
            cv_train_datasets, cv_holdout_datasets = k_folds_splitting_by_signatures(
                self.model_input_x,
                None,
                self.rule_matches_z,
                partitions=self.trainer_config.partitions,
                num_folds=self.trainer_config.cv_n_folds,
                seed=self.trainer_config.seed,
                other_class_id=self.trainer_config.other_class_id,
                other_coeff=self.trainer_config.other_coeff,
                verbose=self.trainer_config.verbose
            )
        else:
            raise ValueError("Implement other psa calculation methods!")

        psx_fs = np.zeros((len(self.model_input_x), self.trainer_config.output_classes))
        psx_w = np.zeros((len(self.model_input_x), self.trainer_config.output_classes))

        for n_fold, (train_dataset, test_dataset) in enumerate(zip(cv_train_datasets, cv_holdout_datasets)):
            merged_set = WrenchDataset(
                examples=train_dataset.examples + test_dataset.examples,
                features=np.vstack((train_dataset.features, test_dataset.features)),
                id2label=train_dataset.id2label,
                ids=train_dataset.ids + test_dataset.ids,
                labels=train_dataset.labels + test_dataset.labels,
                n_class=train_dataset.n_class,
                n_lf=train_dataset.n_lf,
                path=train_dataset.path,
                split=train_dataset.split,
                weak_labels=train_dataset.weak_labels + [[-1] * test_dataset.n_lf] * len(test_dataset)
            )
            hold_out_ids = test_dataset.ids

            ##############
            ##############

            print("Fitting the first stage of Cosine model for CV fold training...")
            self.model = copy.deepcopy(start_model)
            self.model.fit_fs(dataset_train=merged_set, dataset_valid=self.dev_model_input_x, y_train=noisy_y_train_fs,
                              device=self.trainer_config.device, metric=self.trainer_config.target,
                              patience=self.trainer_config.patience,
                              evaluation_step=self.trainer_config.evaluation_step,
                              seed=self.trainer_config.seed)
            pred_labels = self.model.predict_proba(test_dataset)
            for idx in hold_out_ids:
                if idx not in non_labeled_samples:
                    assert psx_fs[idx][0] == 0
                    assert psx_fs[idx][1] == 0
            psx_fs[hold_out_ids] = pred_labels
            print(f"Fold {n_fold + 1} out of {len(cv_train_datasets)} is done")

            ##############
            ##############

            print("Fitting the whole Cosine model for CV fold training...")
            self.model = copy.deepcopy(start_model)
            self.model.fit(dataset_train=merged_set, dataset_valid=self.dev_model_input_x, y_train=noisy_y_train_w,
                           device=self.trainer_config.device, metric=self.trainer_config.target,
                           patience=self.trainer_config.patience, evaluation_step=self.trainer_config.evaluation_step,
                           seed=self.trainer_config.seed)
            pred_labels = self.model.predict_proba(test_dataset)
            for idx in hold_out_ids:
                if idx not in non_labeled_samples:
                    assert psx_w[idx][0] == 0
                    assert psx_w[idx][1] == 0
            psx_w[hold_out_ids] = pred_labels
            print(f"Fold {n_fold + 1} out of {len(cv_train_datasets)} is done")

        # calculate threshold values
        thresholds_fs = calculate_threshold(psx_fs, noisy_y_train_fs, self.trainer_config.output_classes)
        thresholds_w = calculate_threshold(psx_w, noisy_y_train_w, self.trainer_config.output_classes)

        # new_samples_without_matches = []
        # for idx in samples_without_matches:
        #     new_class = int(np.argmax(psx[idx, :]))
        #     if psx[idx, new_class] > thresholds[new_class]:
        #         labels_to_subst[idx] = int(np.argmax(psx[idx, :]))
        #     else:
        #         new_samples_without_matches.append(idx)
        # logger.info(
        #     f"Number of other samples that changed labels: {len(labels_to_subst)} out of {len(samples_without_matches)}"
        # )

        # calculate a noise matrix in advance if applicable
        noise_matrix_fs, inv_noise_matrix_fs, confident_joint_fs = calculate_noise_matrix(
            psx_fs, self.rule_matches_z, thresholds_fs,
            num_classes=self.trainer_config.output_classes,
            noise_matrix=self.trainer_config.noise_matrix,
            calibrate=self.trainer_config.calibrate_cj_matrix
        )

        # calculate a noise matrix in advance if applicable
        noise_matrix_w, inv_noise_matrix_w, confident_joint_w = calculate_noise_matrix(
            psx_w, self.rule_matches_z, thresholds_w,
            num_classes=self.trainer_config.output_classes,
            noise_matrix=self.trainer_config.noise_matrix,
            calibrate=self.trainer_config.calibrate_cj_matrix
        )

        if self.trainer_config.use_prior:
            # normalized((t matrix * prior) + confident_joint)
            updated_t_matrix_fs = update_t_matrix_with_prior(
                confident_joint_fs, self.mapping_rules_labels_t, verbose=self.trainer_config.verbose
            )
            updated_t_matrix_w = update_t_matrix_with_prior(
                confident_joint_w, self.mapping_rules_labels_t, verbose=self.trainer_config.verbose
            )
        else:
            # p * (normalized confident_joint) + (1 - p) * (t matrix)
            updated_t_matrix_fs = update_t_matrix(
                confident_joint_fs, self.mapping_rules_labels_t, self.trainer_config.p,
                verbose=self.trainer_config.verbose
            )
            updated_t_matrix_w = update_t_matrix(
                confident_joint_w, self.mapping_rules_labels_t, self.trainer_config.p,
                verbose=self.trainer_config.verbose
            )
        # todo: clarify with labels_to_subst - replace labels in other samples?
        return updated_t_matrix_fs, updated_t_matrix_w  # , new_samples_without_matches

    def get_updated_labels(
            self, noisy_y_train: np.ndarray, labels_to_subst=None
    ):
        """
        This function recalculates the labels based on the updated t matrix.
        :param noisy_y_train:
        :param labels_to_subst:
        :return:
        """
        # todo: check whether any filtering happened (actually shouldn't....)
        soft_labels = z_t_matrices_to_majority_vote_probs(self.rule_matches_z, self.mapping_rules_labels_t)
        # hard_labels = np.apply_along_axis(probabilities_to_majority_vote, axis=1, arr=soft_labels)
        hard_labels = probs_to_preds(soft_labels)

        if labels_to_subst is not None:
            for c_id, label in labels_to_subst.items():
                hard_labels[c_id] = label

        num_labels_updated = sum(~np.equal(noisy_y_train, hard_labels))
        updated_samples_idx = np.where(~np.equal(noisy_y_train, hard_labels))[0]
        logger.info(f"Labels changed: {num_labels_updated} out of {len(noisy_y_train)}")
        return hard_labels, soft_labels, num_labels_updated, updated_samples_idx

    def initialise_optimizer(self, custom_opt):
        try:
            return custom_opt(params=self.model.parameters(), lr=self.trainer_config.lr)
        except TypeError:
            logger.info(
                "Wrong optimizer parameters. Optimizer should belong to torch.optim class or be PyTorch compatible."
            )
