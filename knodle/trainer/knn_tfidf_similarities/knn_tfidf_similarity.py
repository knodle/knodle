import os
import logging

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import torch
from torch.nn import Module
from torch.utils.data import TensorDataset

from knodle.transformation.majority import input_to_majority_vote_input
from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.trainer import Trainer
from knodle.trainer.knn_tfidf_similarities.knn_config import KNNConfig
from knodle.trainer.utils import log_section
from knodle.trainer.utils.denoise import activate_neighbors
from knodle.trainer.utils.utils import accuracy_of_probs

logger = logging.getLogger(__name__)


class KnnTfidfSimilarity(Trainer):
    def __init__(
            self,
            model: Module,
            mapping_rules_labels_t: np.ndarray,
            model_input_x: TensorDataset,
            rule_matches_z: np.ndarray,
            dev_rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None,
            trainer_config: KNNConfig = None
    ):
        self.tfidf_values = csr_matrix(model_input_x.tensors[0].numpy())
        self.tfidf_values = csr_matrix(model_input_x.tensors[0].numpy())
        self.dev_rule_matches_z = dev_rule_matches_z
        self.dev_model_input_x = dev_model_input_x

        if trainer_config is None:
            trainer_config = KNNConfig(self.model)

        super().__init__(
            model, mapping_rules_labels_t, model_input_x, rule_matches_z, trainer_config
        )

    def train(self):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """

        denoised_rule_matches_z = self._denoise_rule_matches()

        model_input_x, label_probs = input_to_majority_vote_input(
            self.model_input_x, denoised_rule_matches_z, self.mapping_rules_labels_t,
            filter_empty_z_rows=self.trainer_config.filter_non_labelled
        )

        feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        if self.dev_rule_matches_z is not None:
            model_input_x, label_probs = input_to_majority_vote_input(
                self.dev_model_input_x, self.dev_rule_matches_z, self.mapping_rules_labels_t,
                filter_empty_z_rows=self.trainer_config.filter_non_labelled
            )

            dev_feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
            dev_feature_label_dataloader = self._make_dataloader(dev_feature_label_dataset)

        log_section("Training starts", logger)

        self.model.train()
        for current_epoch in range(self.trainer_config.epochs):
            epoch_loss, epoch_acc = 0.0, 0.0
            logger.info("Epoch: {}".format(current_epoch))

            for step, (feature_batch, label_batch) in enumerate(feature_label_dataloader):
                feature_batch = feature_batch.to(self.trainer_config.device)
                label_batch = label_batch.to(self.trainer_config.device)

                self.model.zero_grad()
                predictions = self.model(feature_batch)
                loss = self.trainer_config.criterion(predictions, label_batch)

                loss.backward()
                self.trainer_config.optimizer.step()

                acc = accuracy_of_probs(predictions, label_batch)
                epoch_loss += loss.detach()
                epoch_acc += acc.item()

            avg_loss = epoch_loss / len(feature_label_dataloader)
            avg_acc = epoch_acc / len(feature_label_dataloader)

            log_section(
                "Training Stats",
                logger,
                {"Epoch_accuracy": avg_acc, "Epoch_loss": avg_loss},
            )

            if self.dev_rule_matches_z is not None:
                val_loss, val_acc = self.validation(dev_feature_label_dataloader)
                log_section(
                    "Validation Stats", logger, {"Accuracy": val_acc, "Loss": val_loss}
                )

        log_section("Training done", logger)

    def validation(self, validation_dataloader):
        epoch_loss, epoch_acc = 0.0, 0.0
        self.model.eval()

        with torch.no_grad():
            for feature_batch, label_batch in validation_dataloader:
                feature_batch = feature_batch.to(self.trainer_config.device)
                label_batch = label_batch.to(self.trainer_config.device)
                predictions = self.model(feature_batch)

                loss = self.trainer_config.criterion(predictions, label_batch)
                acc = accuracy_of_probs(predictions, label_batch)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(validation_dataloader), epoch_acc / len(
            validation_dataloader
        )

    def _denoise_rule_matches(self) -> np.ndarray:
        """
        Denoises the applied weak supervision source.
        Args:
            rule_matches_z: Matrix with all applied weak supervision sources. Shape: (Instances x Rules)
        Returns: Denoised / Improved applied labeling function matrix. Shape: (Instances x Rules)
        """

        # load cached data, if available
        cache_dir = self.trainer_config.caching_folder
        if cache_dir is not None:
            cache_file = os.path.join(cache_dir, "denoised_rule_matches_z.lib")
            if os.path.isfile(cache_file):
                return joblib.load(cache_file)

        k = self.trainer_config.k
        if k == 1:
            return self.rule_matches_z

        logger.info(f"Start denoising labeling functions with k: {k}.")

        # Set up data structure, to quickly find nearest neighbors
        if k is not None:
            neighbors = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(self.tfidf_values)
            distances, indices = neighbors.kneighbors(self.tfidf_values, n_neighbors=k)
        else:
            neighbors = NearestNeighbors(radius=self.trainer_config.radius, n_jobs=-1).fit(self.tfidf_values)
            distances, indices = neighbors.radius_neighbors(self.tfidf_values)

        # activate matches.
        denoised_rule_matches_z = activate_neighbors(self.rule_matches_z, indices)

        # save data for caching
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            joblib.dump(cache_file, denoised_rule_matches_z)

        return denoised_rule_matches_z

    def print_step_update(self, step: int, max_steps: int):
        if step % 40 == 0 and not step == 0:
            logger.info(f"  Batch {step}  of  {max_steps}.")
