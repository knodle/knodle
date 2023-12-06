import copy
import logging
import os
from typing import Dict

import numpy as np
from snorkel.utils import probs_to_preds
from torch.nn import Module
from torch.utils.data import TensorDataset

from examples.trainer.ulf.utils import WrenchDataset
from knodle.trainer import MajorityVoteTrainer
from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.cleanlab.utils import calculate_threshold
from knodle.trainer.ulf.config import UlfConfig
from knodle.trainer.ulf.noisy_matrix_estimation import calculate_noise_matrix
from knodle.trainer.ulf.utils import update_t_matrix, assert_psx_empty, collect_stat_updated
from knodle.trainer.utils.split import k_folds_splitting_by_signatures
from knodle.transformation.majority import z_t_matrices_to_probs, input_to_majority_vote_input

logger = logging.getLogger(__name__)

best_of_the_best = 0


@AutoTrainer.register('ulf')
class UlfTrainer(MajorityVoteTrainer):

    def __init__(
            self,
            psx_model: Module = None,
            psx_model_input_x: TensorDataset = None,
            **kwargs):

        self.psx_model = psx_model if psx_model else kwargs.get("model")
        self.psx_model_input_x = psx_model_input_x if psx_model_input_x else kwargs.get("model_input_x")

        self.best_metric_cosine, self.best_metric_ft = 0, 0
        self.best_set_cosine, self.best_set_ft = "", ""

        if kwargs.get("trainer_config", None) is None:
            kwargs["trainer_config"] = UlfConfig()
        super().__init__(**kwargs)

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None,
            test_data=None, target=None, soft=True
    ):
        results = {"CV_finetune": {}, "CV_cosine": {}}
        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)

        # calculate original noisy labels
        self.model_input_x, noisy_y_train, self.rule_matches_z = input_to_majority_vote_input(
            self.rule_matches_z, self.mapping_rules_labels_t, self.model_input_x,
            probability_threshold=self.trainer_config.probability_threshold,
            unmatched_strategy=self.trainer_config.unmatched_strategy,
            ties_strategy=self.trainer_config.ties_strategy,
            use_probabilistic_labels=self.trainer_config.use_probabilistic_labels,
            other_class_id=self.trainer_config.other_class_id
        )

        # save original model and t matrix
        start_model = copy.deepcopy(self.model)
        start_t = copy.deepcopy(self.mapping_rules_labels_t)

        for cv_cosine in [False]:
            # calculate cross-validation OOS probs with cosine and simple fine-tuning in turn

            labels = noisy_y_train

            # set best metrics to 0
            self.best_metric_cosine, self.best_metric_ft = 0, 0
            self.best_set_cosine, self.best_set_ft = "", ""

            # initialize t matrix from the original one
            self.mapping_rules_labels_t = copy.deepcopy(start_t)

            for i in range(self.trainer_config.iterations):
                logger.info(f"Iteration: {i}")

                # denoise the t matrix with either cosine or simple fine-tuned bert-based model
                updated_t_matrix = self.denoise_t_matrix(labels, start_model, if_cosine=cv_cosine)

                # update t matrix
                self.mapping_rules_labels_t = updated_t_matrix

                # calculate the updated labels
                y_hard, y_soft = self.get_updated_labels()
                collect_stat_updated(y_hard, labels)  # calculate how many labels have been changed

                # train end model (to collect the numbers as if self.trainer_config.iterations = i)
                metrics = self.train_end_model(y_hard, y_soft, start_model, test_data, i, target, cv_cosine=cv_cosine)

                # add the results to the final dictionary
                if cv_cosine:
                    results["CV_cosine"][f"Iteration {i}"] = metrics
                else:
                    results["CV_finetune"][f"Iteration {i}"] = metrics

                # use recalculated labels further on
                labels = y_hard

            # add the setting to the dictionary as well
            if cv_cosine is True:
                results["CV_cosine"]["setting"] = {"best_set_cosine_cosine": self.best_set_cosine,
                                                   "best_set_cosine_ft": self.best_set_ft,
                                                   "best_valid_score_cosine_cosine": self.best_metric_cosine,
                                                   "best_valid_score_cosine_ft": self.best_metric_ft}
            else:
                results["CV_finetune"]["setting"] = {"best_set_ft_cosine": self.best_set_cosine,
                                                     "best_set_ft_ft": self.best_set_ft,
                                                     "best_valid_score_ft_cosine": self.best_metric_cosine,
                                                     "best_valid_score_ft_ft": self.best_metric_ft}

        return results

    def denoise_t_matrix(self, noisy_y_train: np.ndarray, start_model, if_cosine: bool) -> np.ndarray:
        """
        Recalculation of labeling functions to labels matrix
        :param noisy_y_train: original t (labeling functions x labels) matrix
        :param start_model: raw untrained model which will be copied in each cross-validation batch training
        :return: updated labeling functions x labels matrix
        """

        non_labeled_samples = list(np.where(np.all(self.rule_matches_z == 0, axis=1))[0])

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
        psx = np.zeros((len(self.model_input_x), self.trainer_config.output_classes))

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
            assert_psx_empty(psx, hold_out_ids, non_labeled_samples)

            self.model = copy.deepcopy(start_model)

            if if_cosine:
                # train the model on cross-validation folds with cosine
                print("Fitting the whole Cosine model for CV fold training...")
                self.model.fit(dataset_train=merged_set, dataset_valid=self.dev_model_input_x,
                               y_train=noisy_y_train,
                               device=self.trainer_config.device, metric=self.trainer_config.target,
                               patience=self.trainer_config.patience,
                               evaluation_step=self.trainer_config.evaluation_step,
                               seed=self.trainer_config.seed)
            else:
                # train the model on cross-validation folds with simple fine-tuned bert-based model
                print("Fitting the BERT model only for CV fold training...")
                self.model.fit_fs(dataset_train=merged_set, dataset_valid=self.dev_model_input_x,
                                  y_train=noisy_y_train,
                                  device=self.trainer_config.device, metric=self.trainer_config.target,
                                  patience=self.trainer_config.patience,
                                  evaluation_step=self.trainer_config.evaluation_step,
                                  seed=self.trainer_config.seed)
            # calculate out-of-sample probabilities
            pred_labels = self.model.predict_proba(test_dataset)
            psx[hold_out_ids] = pred_labels
            print(f"Fold {n_fold + 1} out of {len(cv_train_datasets)} is done")

        return self.calculate_t(psx, noisy_y_train)

    def calculate_t(self, psx, noisy_y_train) -> np.ndarray:
        thresholds = calculate_threshold(psx, noisy_y_train, self.trainer_config.output_classes)
        noise_matrix, inv_noise_matrix, confident_joint = calculate_noise_matrix(
            psx, self.rule_matches_z, thresholds,
            num_classes=self.trainer_config.output_classes,
            noise_matrix=self.trainer_config.noise_matrix,
            calibrate=self.trainer_config.calibrate_cj_matrix
        )
        updated_t_matrix = update_t_matrix(
            confident_joint, self.mapping_rules_labels_t, self.trainer_config.p,
            verbose=self.trainer_config.verbose
        )
        print(updated_t_matrix)
        return updated_t_matrix

    def train_end_model(
            self, y_hard, y_soft, start_model, test_data, i, target, cv_cosine: bool
    ):
        """ Train end models with updated labels (cosine and fine-tuning) """

        cv = "cosine" if cv_cosine else "ft"

        # run end cosine model
        metrics = {"END_finetune": {}, "END_cosine": {}}

        # full cosine end model with hard labels
        metrics = self.train_end_cosine(start_model, test_data, y_hard, target, metrics, False, i, cv)

        # full cosine end model with soft labels
        metrics = self.train_end_cosine(start_model, test_data, y_soft, target, metrics, True, i, cv)

        # first stage cosine with hard labels
        metrics = self.train_end_fs(start_model, test_data, y_hard, target, metrics, False, i, cv)

        # first stage cosine with soft labels
        metrics = self.train_end_fs(start_model, test_data, y_soft, target, metrics, True, i, cv)

        logger.info(f"Iteration {i}, results: {str(metrics)}")

        return metrics

    def train_end_cosine(self, model, test_data, labels, target, metrics, if_soft, i, cv) -> Dict:
        # copy the model from the original state
        self.model = copy.deepcopy(model)

        # use either soft or hard labels
        modus = "soft" if if_soft else "hard"
        logger.info(f"END MODEL: Train full Cosine model with {modus} labels...")

        # train the model and obtain the validation and test metrics
        curr_metrics = self.train_n_get_metrics_cosine(test_data, labels, if_soft=if_soft)

        logger.info(f"Full Cosine model. {modus} labels. "
                    f"Iteration {i + 1} out of {self.trainer_config.iterations} done."
                    f"Target valid metric {target}: {curr_metrics[f'valid {target}']}. "
                    f"Target test metric {target}: {curr_metrics[f'test {target}']}")
        metrics["END_cosine"][modus] = curr_metrics  # add the metrics to the global dictionary

        # if there is a better model, save it
        self.best_metric_cosine, if_best = self.check_metric_value(
            curr_metrics, self.best_metric_cosine, target, f"{self.trainer_config.save_model_name}_cv_{cv}_end_cosine"
        )
        # update the best score as well
        self.best_set_cosine = f"Iteration_{i}_{modus}_label" if if_best else self.best_set_cosine

        return metrics

    def train_end_fs(self, model, data, labels, target, metrics, if_soft, i, cv):
        # copy the model from the original state
        self.model = copy.deepcopy(model)

        # use either soft or hard labels
        modus = "soft" if if_soft else "hard"
        logger.info(f"END MODEL: Train only first stage of Cosine model (= pretraining) with {modus} labels...")

        # train the model and obtain the validation and test metrics
        curr_metrics = self.train_n_get_metrics_pretraining(data, labels, if_soft=if_soft)

        print(f"First stage of Cosine model. {modus} labels. "
              f"Iteration {i + 1} out of {self.trainer_config.iterations} done."
              f"Target valid metric {target}: {curr_metrics[f'valid {target}']}. "
              f"Target test metric {target}: {curr_metrics[f'test {target}']}")

        metrics["END_finetune"][modus] = curr_metrics  # add the metrics to the global dictionary

        # if there is a better model, save it
        self.best_metric_ft, if_best = self.check_metric_value(
            curr_metrics, self.best_metric_ft, target, f"{self.trainer_config.save_model_name}_cv_{cv}_end_ft"
        )
        # update the best score as well
        self.best_set_ft = f"Iteration_{i}_{modus}_label" if if_best else self.best_set_ft

        return metrics

    def check_metric_value(self, metrics, best, target, model_name):
        print(metrics)
        try:
            if metrics[f'valid {target}'] > best:
                self.model.save(os.path.join(self.trainer_config.save_model_path, model_name))
                print(f"Old metric: {best} \t New best metric: {metrics[f'valid {target}']}. \t "
                      f"New model is saved to {os.path.join(self.trainer_config.save_model_path, model_name)}")
                return metrics[f'valid {target}'], True

            else:
                print(f"The current metric {metrics[f'valid {target}']} is worse than the best one {best}; "
                      f"no model is saved.")
                return best, False
        except KeyError:
            print("Please! Check target metric!")

    def train_n_get_metrics_cosine(self, test_data, noisy_y_train, if_soft):
        """ Train the cosine end model """
        print("Fitting the whole cosine....")
        self.model.fit(
            dataset_train=self.model_input_x, dataset_valid=self.dev_model_input_x, y_train=noisy_y_train,
            soft_labels=if_soft, device=self.trainer_config.device, metric=self.trainer_config.target,
            patience=self.trainer_config.patience, evaluation_step=self.trainer_config.evaluation_step,
            seed=self.trainer_config.seed
        )
        # print(history)
        # if len(list(history["selftrain"].keys())) > 0:
        #     dev_metric = history["selftrain"][list(history["selftrain"])[-1]][f"best_val_{self.trainer_config.target}"]
        # else:
        #     warnings.warn("Alarm! No self-training by cosine!")
        #     dev_metric = history["pretrain"][list(history["pretrain"])[-1]][f"best_val_{self.trainer_config.target}"]
        return self.get_valid_test_scores(test_data)

    def train_n_get_metrics_pretraining(self, test_data, noisy_y_train, if_soft):
        """ Train the simple pretrained bert-based end model """
        print("Fitting the first part of cosine (pretraining)....")
        self.model.fit_fs(
            dataset_train=self.model_input_x, dataset_valid=self.dev_model_input_x, y_train=noisy_y_train,
            soft_labels=if_soft, device=self.trainer_config.device, metric=self.trainer_config.target,
            patience=self.trainer_config.patience, evaluation_step=self.trainer_config.evaluation_step,
            seed=self.trainer_config.seed
        )
        return self.get_valid_test_scores(test_data)

    def get_valid_test_scores(self, test_data) -> Dict:
        val_metric = self.model.test(self.dev_model_input_x, self.trainer_config.target)  # evaluate on the dev set
        test_metric = self.model.test(test_data,
                                      self.trainer_config.target)  # test on the test set - jsut to collect the results

        metrics = {f"valid {self.trainer_config.target}": val_metric, f"test {self.trainer_config.target}": test_metric}
        print(str(metrics))

        return metrics

    def get_updated_labels(self, labels_to_subst=None):
        """ This function recalculates the labels based on the updated t matrix. """
        soft_labels = z_t_matrices_to_probs(self.rule_matches_z, self.mapping_rules_labels_t)
        hard_labels = probs_to_preds(soft_labels)

        if labels_to_subst is not None:
            for c_id, label in labels_to_subst.items():
                hard_labels[c_id] = label
        return hard_labels, soft_labels

    def initialise_optimizer(self, custom_opt):
        try:
            return custom_opt(params=self.model.parameters(), lr=self.trainer_config.lr)
        except TypeError:
            logger.info(
                "Wrong optimizer parameters. Optimizer should belong to torch.optim class or be PyTorch compatible."
            )
