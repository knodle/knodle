import copy
import logging
import os

import torch
from torch.utils.data import DataLoader
from joblib import dump

from knodle.trainer.baseline.majority import MajorityVoteTrainer
from knodle.trainer.utils import log_section
from knodle.trainer.wscrossweigh.data_splitting_by_rules import k_folds_splitting_by_rules
from knodle.transformation.filter import filter_empty_probabilities
from knodle.transformation.majority import z_t_matrices_to_majority_vote_probs

logger = logging.getLogger(__name__)
torch.set_printoptions(edgeitems=100)


class WSCrossWeighWeightsCalculator(MajorityVoteTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # save the copy of the original model; later wscrossweigh models for each training with a new hold-out fold
        # will be copied from it
        self.wscrossweigh_model = copy.deepcopy(self.model).to(self.trainer_config.device)
        self.sample_weights = torch.empty(0)

    def calculate_weights(self) -> torch.FloatTensor:
        """
        This function calculates the sample_weights for samples using WSCrossWeigh method
        :return matrix of the sample sample_weights
        """

        # initialize optimizer
        self.trainer_config.optimizer = self.initialise_optimizer()

        if self.trainer_config.folds < 2:
            raise ValueError("Number of folds should be at least 2 to perform WSCrossWeigh denoising")

        logger.info("======= Denoising with WSCrossWeigh is started =======")
        os.makedirs(self.trainer_config.caching_folder, exist_ok=True)

        noisy_y_train = z_t_matrices_to_majority_vote_probs(
            self.rule_matches_z, self.mapping_rules_labels_t, self.trainer_config.other_class_id
        )

        if self.trainer_config.filter_non_labelled:
            self.model_input_x, noisy_y_train, self.rule_matches_z = filter_empty_probabilities(
                self.model_input_x, noisy_y_train, self.rule_matches_z
            )

        # initialise sample weights
        self.sample_weights = self.initialise_sample_weights()

        train_datasets, test_datasets = \
            k_folds_splitting_by_rules(
                self.model_input_x,
                noisy_y_train,
                self.rule_matches_z,
                self.trainer_config.partitions,
                self.trainer_config.folds,
                self.trainer_config.other_class_id
            )

        for iter, (train_dataset, test_dataset) in enumerate(zip(train_datasets, test_datasets)):
            log_section(
                f"WSCrossWeigh Iteration {iter + 1}/{self.trainer_config.partitions * self.trainer_config.folds}:",
                logger
            )

            # for each fold the model is trained from scratch
            self.model = copy.deepcopy(self.wscrossweigh_model).to(self.trainer_config.device)
            test_loader = self._make_dataloader(test_dataset)
            train_loader = self._make_dataloader(train_dataset)
            self._train_loop(train_loader)
            self.cw_test(test_loader)

            log_section(f"WSCrossWeigh Partition {iter + 1} is done", logger)

        dump(self.sample_weights, os.path.join(
            self.trainer_config.caching_folder, f"sample_weights_{self.trainer_config.caching_suffix}.lib"))

        logger.info("======= Denoising with WSCrossWeigh is completed =======")
        return self.sample_weights

    def cw_test(self, test_loader: DataLoader) -> None:
        """
        This function tests of trained WSCrossWeigh model on a hold-out fold, compared the predicted labels with the
        ones got with weak supervision and reduces sample_weights of disagreed samples
        :param test_loader: loader with the data which is used for testing (hold-out fold)
        """
        self.wscrossweigh_model.eval()
        correct_predictions, wrong_predictions = 0, 0

        with torch.no_grad():
            for batch in test_loader:
                features, labels = self._load_batch(batch)
                data_features, data_indices = features[:-1], features[-1]

                outputs = self.wscrossweigh_model(*data_features)
                outputs = outputs[0] if not isinstance(outputs, torch.Tensor) else outputs
                _, predicted = torch.max(outputs.data, -1)
                predictions = predicted.tolist()

                for curr_pred in range(len(predictions)):
                    gold = labels.tolist()[curr_pred]
                    gold_classes = gold.index(max(gold))
                    guess = predictions[curr_pred]
                    if guess != gold_classes:       # todo: what if more than one class could be predicted? e.g. conll
                        wrong_predictions += 1
                        curr_id = data_indices[curr_pred].tolist()
                        self.sample_weights[curr_id] *= self.trainer_config.weight_reducing_rate
                    else:
                        correct_predictions += 1
        logger.info("Correct predictions: {:.3f}%, wrong predictions: {:.3f}%".format(
            correct_predictions * 100 / (correct_predictions + wrong_predictions),
            wrong_predictions * 100 / (correct_predictions + wrong_predictions)))

    def initialise_sample_weights(self) -> torch.FloatTensor:
        """ Initialise a sample_weights matrix (num_samples x 1): weights for all samples equal sample_start_weights """
        return torch.FloatTensor([self.trainer_config.samples_start_weights] * self.model_input_x.tensors[0].shape[0])
