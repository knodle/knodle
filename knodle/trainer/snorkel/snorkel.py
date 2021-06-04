from typing import Tuple

import numpy as np
from snorkel.labeling.model import LabelModel

from torch.optim import SGD
from torch.utils.data import TensorDataset

from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.baseline.majority import MajorityVoteTrainer
from knodle.trainer.knn_denoising.knn import KnnDenoisingTrainer

from knodle.trainer.snorkel.config import SnorkelConfig, SnorkelKNNConfig
from knodle.trainer.snorkel.utils import (
    z_t_matrix_to_snorkel_matrix,
    prepare_empty_rule_matches,
    add_labels_for_empty_examples)
from knodle.transformation.filter import filter_tensor_dataset_by_indices

@AutoTrainer.register('snorkel')
class SnorkelTrainer(MajorityVoteTrainer):
    """Provides a wrapper around the Snorkel system. See https://github.com/snorkel-team/snorkel for more details.
    Formally, a generative model P(Y, Y') is learned, with Y' = Z * T, followed by a discriminative model specified
    by the user.
    """

    def __init__(self, **kwargs):
        if kwargs.get("trainer_config", None) is None:
            kwargs["trainer_config"] = SnorkelConfig(optimizer=SGD, lr=0.001)
        super().__init__(**kwargs)

    def _snorkel_denoising(
            self, model_input_x: TensorDataset, rule_matches_z: np.ndarray
    ) -> Tuple[TensorDataset, np.ndarray]:
        """
		Trains the generative model.
        Premise:
            Snorkel can not make use of rule-unlabeled examples (no rule matches).
            The generative LabelModel assigns such examples a uniform distribution over all available labels,
            which contradicts the desired behaviour. Such examples should be either filtered or assigned an
            "other class id".
        Filtering / other class strategy:
            filter_non_labelled = True:
                Drop the unlabeled examples completely prior to LabelModel.
            filter_non_labelled = False:
                However, we might want to keep negative examples in the training data, but we should not pass them
                through the LabelModel. Therefore, the rule-unlabeled part of the data skips the LabelModel step
                and is added directly to the output data with manually assigned "other class id".

        Args:
            model_input_x: feature input to the classifier
            rule_matches_z: input rule matches

        Returns:
            eventually filtered model input,
            corresponding probability distributions over labels generated by Snorkel. Shape: (#Instances x #Labels)
        """

        # initialise optimizer
        self.trainer_config.optimizer = self.initialise_optimizer()

        # create Snorkel matrix
        non_empty_mask, rule_matches_z = prepare_empty_rule_matches(rule_matches_z)
        L_train = z_t_matrix_to_snorkel_matrix(rule_matches_z, self.mapping_rules_labels_t)

        # train LabelModel
        label_model = LabelModel(cardinality=self.mapping_rules_labels_t.shape[1], verbose=True)
        label_model.fit(
            L_train,
            n_epochs=self.trainer_config.label_model_num_epochs,
            log_freq=self.trainer_config.label_model_log_freq,
            seed=self.trainer_config.seed
        )
        label_probs_gen = label_model.predict_proba(L_train)

        if self.trainer_config.filter_non_labelled:
            # filter out respective input irrevocably from all data
            model_input_x = filter_tensor_dataset_by_indices(dataset=model_input_x, filter_ids=non_empty_mask)
            label_probs = label_probs_gen
        else:
            # add "other class" labels for empty examples in model input x, that were not given to Snorkel
            label_probs = add_labels_for_empty_examples(
                label_probs_gen=label_probs_gen, non_zero_mask=non_empty_mask,
                output_classes=self.trainer_config.output_classes,
                other_class_id=self.trainer_config.other_class_id)
        return model_input_x, label_probs

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ):
        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)

        model_input_x, label_probs = self._snorkel_denoising(self.model_input_x, self.rule_matches_z)

        # Standard training
        feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self._train_loop(feature_label_dataloader)


@AutoTrainer.register('snorkel_knn')
class SnorkelKNNDenoisingTrainer(SnorkelTrainer, KnnDenoisingTrainer):
    """Calls k-NN denoising, before the Snorkel generative and discriminative training is started.
    """
    def __init__(self, **kwargs):
        if kwargs.get("trainer_config", None) is None:
            kwargs["trainer_config"] = SnorkelKNNConfig(optimizer=SGD, lr=0.001)
        super().__init__(**kwargs)

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ):
        # Snorkel denoising
        denoised_rule_matches_z = self._knn_denoise_rule_matches()
        model_input_x, label_probs = self._snorkel_denoising(self.model_input_x, denoised_rule_matches_z)

        # Standard training
        feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self._train_loop(feature_label_dataloader)
