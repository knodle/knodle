import logging
import os
from copy import copy
from typing import Dict

import numpy as np
import torch
from joblib import load
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import TensorDataset

from knodle.trainer.baseline.no_denoising import NoDenoisingTrainer
from knodle.trainer.crossweigh_weighing.config import DSCrossWeighDenoisingConfig
from knodle.trainer.crossweigh_weighing.crossweigh_weights_calculator import DSCrossWeighWeightsCalculator

from knodle.transformation.filter import filter_empty_probabilities
from knodle.transformation.majority import z_t_matrices_to_majority_vote_probs
from knodle.transformation.torch_input import input_info_labels_to_tensordataset

torch.set_printoptions(edgeitems=100)
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True


class DSCrossWeighTrainer(NoDenoisingTrainer):

    def __init__(
            self,
            cw_model: Module = None,
            cw_model_input_x: TensorDataset = None,
            cw_rule_matches_z: np.ndarray = None,
            evaluation_method: str = "sklearn_classification_report",
            dev_labels_ids: Dict = None,
            path_to_weights: str = "data",
            run_classifier: bool = True,
            use_weights: bool = True,
            **kwargs
    ):
        self.cw_model = cw_model if cw_model else kwargs.get("model")
        self.cw_model_input_x = cw_model_input_x if cw_model_input_x else kwargs.get("model_input_x")
        self.cw_rule_matches_z = cw_rule_matches_z if cw_rule_matches_z else kwargs.get("rule_matches_z")

        if kwargs.get("trainer_config") is None:
            kwargs["trainer_config"] = DSCrossWeighDenoisingConfig(
                optimizer=SGD(kwargs.get("model").parameters(), lr=0.001),
                cw_optimizer=SGD(self.cw_model.parameters(), lr=0.001)
            )
        super().__init__(**kwargs)

        self.evaluation_method = evaluation_method
        self.dev_labels_ids = dev_labels_ids
        self.path_to_weights = path_to_weights
        self.run_classifier = run_classifier
        self.use_weights = use_weights

        logger.info("CrossWeigh Config is used: {}".format(self.trainer_config.__dict__))

    def train(self):
        """ This function sample_weights the samples with DSCrossWeigh method and train the model """

        train_labels = self.calculate_labels()

        sample_weights = self._get_sample_weights() if self.use_weights \
            else torch.FloatTensor([1] * len(self.model_input_x))

        if not self.run_classifier:
            logger.info("No classifier is to be trained")
            return
        logger.info("Classifier training is started")

        train_loader = self._make_dataloader(
            input_info_labels_to_tensordataset(self.model_input_x, sample_weights.cpu().numpy(), train_labels)
        )

        self.train_loop(train_loader, use_sample_weights=True, draw_plot=True)

    def calculate_labels(self) -> np.ndarray:
        """ This function calculates label probabilities and filter out non labelled samples, when needed """
        if not self.trainer_config.filter_non_labelled and self.trainer_config.other_class_id is None:
            self.trainer_config.other_class_id = self.mapping_rules_labels_t.shape[1]

        train_labels = z_t_matrices_to_majority_vote_probs(
            self.rule_matches_z, self.mapping_rules_labels_t, self.trainer_config.other_class_id
        )

        if self.trainer_config.filter_non_labelled:
            self.model_input_x, train_labels, self.rule_matches_z = filter_empty_probabilities(
                self.model_input_x, train_labels, self.rule_matches_z
            )

        if train_labels.shape[1] != self.trainer_config.output_classes:
            raise ValueError(
                f"The number of output classes {self.trainer_config.output_classes} do not correspond to labels "
                f"probabilities dimension {train_labels.shape[1]}"
            )

        return train_labels

    def _get_sample_weights(self) -> torch.FloatTensor:
        """ This function checks whether there are accessible already pretrained sample weights. If yes, return
        them. If not, calculates sample weights calling method of DSCrossWeighWeightsCalculator class"""

        if os.path.isfile(os.path.join(self.path_to_weights, "sample_weights.lib")):
            logger.info("Already pretrained samples sample_weights will be used.")
            sample_weights = load(os.path.join(self.path_to_weights, "sample_weights.lib"))
        else:
            logger.info("No pretrained sample weights are found, they will be calculated now")
            sample_weights = DSCrossWeighWeightsCalculator(
                model=self.cw_model,
                mapping_rules_labels_t=self.mapping_rules_labels_t,
                model_input_x=self.cw_model_input_x,
                rule_matches_z=self.cw_rule_matches_z,
                path_to_weights=self.path_to_weights,
                trainer_config=self.get_denoising_config(),
            ).calculate_weights()
            logger.info(f"Sample weights are calculated and saved to {self.path_to_weights} file")
        return sample_weights

    def get_denoising_config(self):
        """ Get a config for dscrossweigh sample weights calculation """
        weights_calculation_config = copy(self.trainer_config)
        weights_calculation_config.epochs = self.trainer_config.cw_epochs
        weights_calculation_config.optimizer = self.trainer_config.cw_optimizer
        weights_calculation_config.batch_size = self.trainer_config.cw_batch_size
        weights_calculation_config.filter_non_labelled = self.trainer_config.cw_filter_non_labelled
        weights_calculation_config.other_class_id = self.trainer_config.cw_other_class_id
        weights_calculation_config.grad_clipping = self.trainer_config.cw_grad_clipping
        weights_calculation_config.if_set_seed = self.trainer_config.cw_if_set_seed
        return weights_calculation_config

    # def calculate_dev_metrics(self, predictions: np.ndarray, gold_labels: np.ndarray) -> Union[Dict, None]:
    #     """
    #     Returns the dictionary of metrics calculated on the dev set with one of the evaluation functions
    #     or None, if the needed evaluation method was not found
    #     """
    #
    #     if self.evaluation_method == "tacred":
    #         if self.dev_labels_ids is None:
    #             logging.warning(
    #                 "Labels to labels ids correspondence is needed to make TACRED specific evaluation. Since it is "
    #                 "absent now, the standard sklearn metrics will be calculated instead"
    #             )
    #             return classification_report(y_true=gold_labels, y_pred=predictions, output_dict=True)
    #
    #         return calculate_dev_tacred_metrics(predictions, gold_labels, self.dev_labels_ids)
    #
    #     elif self.evaluation_method == "sklearn_classification_report":
    #         return classification_report(y_true=gold_labels, y_pred=predictions, output_dict=True)
    #
    #     else:
    #         logging.warning("No evaluation method is given. The evaluation on dev data is skipped")
    #         return None
