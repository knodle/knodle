import logging
import os
from copy import copy

import numpy as np
import torch
from joblib import load
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import TensorDataset

from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.baseline.majority import MajorityVoteTrainer
from knodle.trainer.wscrossweigh.config import WSCrossWeighConfig
from knodle.trainer.wscrossweigh.wscrossweigh_weights_calculator import WSCrossWeighWeightsCalculator

from knodle.transformation.filter import filter_empty_probabilities
from knodle.transformation.majority import z_t_matrices_to_majority_vote_probs
from knodle.transformation.torch_input import input_info_labels_to_tensordataset

torch.set_printoptions(edgeitems=100)
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True


@AutoTrainer.register('wscrossweigh')
class WSCrossWeighTrainer(MajorityVoteTrainer):

    def __init__(
            self,
            cw_model: Module = None,
            cw_model_input_x: TensorDataset = None,
            cw_rule_matches_z: np.ndarray = None,
            run_classifier: bool = True,  # set to False if you want only the calculation of the sample weights
            use_weights: bool = True,  # set to False if you want to use weights = 1 (baseline)
            **kwargs
    ):
        self.cw_model = cw_model if cw_model else kwargs.get("model")
        self.cw_model_input_x = cw_model_input_x if cw_model_input_x else kwargs.get("model_input_x")
        self.cw_rule_matches_z = cw_rule_matches_z if cw_rule_matches_z else kwargs.get("rule_matches_z")

        if kwargs.get("trainer_config") is None:
            kwargs["trainer_config"] = WSCrossWeighConfig(
                optimizer=SGD,
                cw_optimizer=SGD,
                lr=0.001,
                cw_lr=0.001,
            )
        super().__init__(**kwargs)

        self.run_classifier = run_classifier
        self.use_weights = use_weights

        logger.info("CrossWeigh Config is used: {}".format(self.trainer_config.__dict__))

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ):
        """ This function sample_weights the samples with WSCrossWeigh method and train the model """
        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)
        self._apply_rule_reduction()

        # initialise optimizer
        self.trainer_config.optimizer = self.initialise_optimizer()

        train_labels = self.calculate_labels()

        sample_weights = self._get_sample_weights() if self.use_weights \
            else torch.FloatTensor([1] * len(self.model_input_x))

        if not self.run_classifier:
            logger.info("No classifier is to be trained")
            return
        logger.info("Classifier training is started")

        train_loader = self._make_dataloader(
            input_info_labels_to_tensordataset(self.model_input_x, sample_weights.cpu().detach().numpy(), train_labels)
        )

        self._train_loop(train_loader, use_sample_weights=True, draw_plot=self.trainer_config.draw_plot)

    def calculate_labels(self) -> np.ndarray:
        """ This function calculates label probabilities and filter out non labelled samples, when needed """
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
        them. If not, calculates sample weights calling method of WSCrossWeighWeightsCalculator class"""

        if os.path.isfile(os.path.join(
                self.trainer_config.caching_folder, f"sample_weights_{self.trainer_config.caching_suffix}.lib")
        ):
            logger.info("Already pretrained samples sample_weights will be used.")
            sample_weights = load(os.path.join(
                self.trainer_config.caching_folder, f"sample_weights_{self.trainer_config.caching_suffix}.lib")
            )
        else:
            logger.info("No pretrained sample weights are found, they will be calculated now")
            sample_weights = WSCrossWeighWeightsCalculator(
                model=self.cw_model,
                mapping_rules_labels_t=self.mapping_rules_labels_t,
                model_input_x=self.cw_model_input_x,
                rule_matches_z=self.cw_rule_matches_z,
                trainer_config=self.get_denoising_config(),
            ).calculate_weights()
            logger.info(f"Sample weights are calculated and saved to {self.trainer_config.caching_folder} folder")
        return sample_weights

    def get_denoising_config(self):
        """ Get a config for WSCrossWeigh sample weights calculation """
        weights_calculation_config = copy(self.trainer_config)
        weights_calculation_config.epochs = self.trainer_config.cw_epochs
        weights_calculation_config.optimizer = self.trainer_config.cw_optimizer
        weights_calculation_config.lr = self.trainer_config.cw_lr
        weights_calculation_config.batch_size = self.trainer_config.cw_batch_size
        weights_calculation_config.filter_non_labelled = self.trainer_config.cw_filter_non_labelled
        weights_calculation_config.other_class_id = self.trainer_config.cw_other_class_id
        weights_calculation_config.grad_clipping = self.trainer_config.cw_grad_clipping
        weights_calculation_config.seed = self.trainer_config.cw_seed
        weights_calculation_config.saved_models_dir = None
        return weights_calculation_config
