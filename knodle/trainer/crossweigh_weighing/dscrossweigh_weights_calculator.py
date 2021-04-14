import copy
import logging
import os

import torch
from torch.utils.data import DataLoader
from joblib import dump

from knodle.trainer.baseline.majority import MajorityVoteTrainer
from knodle.trainer.crossweigh_weighing.data_preparation import k_folds_splitting
from knodle.transformation.filter import filter_empty_probabilities
from knodle.transformation.majority import z_t_matrices_to_majority_vote_probs

logger = logging.getLogger(__name__)
torch.set_printoptions(edgeitems=100)


class DSCrossWeighWeightsCalculator(MajorityVoteTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.crossweigh_model = copy.deepcopy(self.model)
        self.sample_weights = torch.empty(0)

    def calculate_weights(self) -> torch.FloatTensor:
        """
        This function calculates the sample_weights for samples using DSCrossWeigh method
        :return matrix of the sample sample_weights
        """

        if self.trainer_config.folds < 2:
            raise ValueError("Number of folds should be at least 2 to perform DSCrossWeigh denoising")

        logger.info("======= Denoising with DSCrossWeigh is started =======")
        os.makedirs(self.trainer_config.caching_folder, exist_ok=True)

        labels = z_t_matrices_to_majority_vote_probs(
            self.rule_matches_z, self.mapping_rules_labels_t, self.trainer_config.other_class_id
        )

        if self.trainer_config.filter_non_labelled:
            self.model_input_x, labels, self.rule_matches_z = filter_empty_probabilities(
                self.model_input_x, labels, self.rule_matches_z
            )

        sample_weights = self.initialise_sample_weights()

        train_datasets, test_datasets = \
            k_folds_splitting(
                self.model_input_x,
                labels,
                self.rule_matches_z,
                self.trainer_config.partitions,
                self.trainer_config.folds,
                self.trainer_config.other_class_id
            )

        for iter, (train_dataset, test_dataset) in enumerate(zip(train_datasets, test_datasets)):
            # for each fold the model is trained from scratch
            self.crossweigh_model = copy.deepcopy(self.model).to(self.trainer_config.device)
            # todo: ???? WTF ??? self.model = copy.deepcopy(self.model).to(self.trainer_config.device) ???
            test_loader = self._make_dataloader(test_dataset)
            train_loader = self._make_dataloader(train_dataset)
            self._train_loop(train_loader)
            self.cw_test(test_loader)

        # other_sample_ids = self._get_other_sample_ids(labels) if self.trainer_config.other_class_id else None
        # rules_samples_ids_dict = self._get_rules_samples_ids_dict()
        #
        # self.sample_weights = self.initialise_sample_weights()
        #
        # for partition in range(self.trainer_config.partitions):
        #     log_section(f"CrossWeigh Partition {partition + 1}/{self.trainer_config.partitions}:", logger)
        #     shuffled_rules_ids = self._get_shuffled_rules_idx()  # shuffle anew for each cw round
        #     for fold in range(self.trainer_config.folds):
        #         # for each fold the model is trained from scratch
        #         self.crossweigh_model = copy.deepcopy(self.model).to(self.trainer_config.device)
        #         train_loader, test_loader = self.get_cw_data(
        #             shuffled_rules_ids, rules_samples_ids_dict, labels, fold, other_sample_ids
        #         )
        #         self._train_loop(train_loader)
        #         self.cw_test(test_loader)
        #     log_section(f"CrossWeigh Partition {partition + 1} is done", logger)

        dump(sample_weights, os.path.join(
            self.trainer_config.caching_folder, f"sample_weights_{self.trainer_config.caching_suffix}.lib"
        ))
        logger.info("======= Denoising with DSCrossWeigh is completed =======")
        return self.sample_weights

    def cw_test(self, test_loader: DataLoader) -> None:
        """
        This function tests of trained DSCrossWeigh model on a hold-out fold, compared the predicted labels with the
        ones got with weak supervision and reduces sample_weights of disagreed samples
        :param test_loader: loader with the data which is used for testing (hold-out fold)
        """
        self.crossweigh_model.eval()
        correct_predictions, wrong_predictions = 0, 0

        with torch.no_grad():
            for batch in test_loader:
                features, labels = self._load_batch(batch)
                data_features, data_indices = features[:-1], features[-1]

                outputs = self.crossweigh_model(*data_features)
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
