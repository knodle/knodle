import logging

import numpy as np
import os
import torch
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset

from knodle.trainer import TrainerConfig
from knodle.trainer.trainer import Trainer
from knodle.trainer.utils import log_section
from knodle.trainer.utils.denoise import get_majority_vote_probs, activate_all_neighbors
from knodle.trainer.utils.utils import accuracy_of_probs, extract_tensor_from_dataset

torch.manual_seed(123)

logger = logging.getLogger(__name__)


class KnnTfidfSimilarity(Trainer):
    def __init__(
            self,
            model: Module,
            mapping_rules_labels_t: np.ndarray,
            model_input_x: TensorDataset,
            rule_matches_z: np.ndarray,
            tfidf_values: csr_matrix,
            k: int,
            dev_rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None,
            trainer_config: TrainerConfig = None,
            cache_denoised_matches=False,
            caching_prefix="knn_cached",
    ):
        super().__init__(
            model, mapping_rules_labels_t, model_input_x, rule_matches_z, trainer_config
        )
        self.tfidf_values = tfidf_values
        self.k = k
        self.cache_denoised_matches = cache_denoised_matches
        self.caching_prefix = caching_prefix
        self.dev_rule_matches_z = dev_rule_matches_z
        self.dev_model_input_x = dev_model_input_x

    def train(self):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """

        if self.cache_denoised_matches:
            denoised_rule_matches_z = self.get_or_create_z(self.caching_prefix, self.k)
        else:
            denoised_rule_matches_z = self._denoise_rule_matches(self.rule_matches_z)

        labels = get_majority_vote_probs(
            denoised_rule_matches_z, self.mapping_rules_labels_t
        )

        labels = Tensor(labels)
        labels = labels.to(self.trainer_config.device)

        model_input_x_tensor = extract_tensor_from_dataset(self.model_input_x, 0)
        model_input_x_tensor = model_input_x_tensor.to(self.trainer_config.device)

        feature_label_dataset = TensorDataset(model_input_x_tensor, labels)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset, True)

        if self.dev_rule_matches_z is not None:
            dev_labels = get_majority_vote_probs(
                self.dev_rule_matches_z, self.mapping_rules_labels_t
            )

            dev_labels = Tensor(dev_labels)
            dev_labels = dev_labels.to(self.trainer_config.device)

            dev_model_input_x_tensor = extract_tensor_from_dataset(
                self.dev_model_input_x, 0
            )
            dev_model_input_x_tensor = dev_model_input_x_tensor.to(
                self.trainer_config.device
            )

            dev_feature_label_dataset = TensorDataset(
                dev_model_input_x_tensor, dev_labels
            )
            dev_feature_label_dataloader = self._make_dataloader(
                dev_feature_label_dataset, True
            )

        log_section("Training starts", logger)

        self.model.train()
        for current_epoch in range(self.trainer_config.epochs):
            epoch_loss, epoch_acc = 0.0, 0.0
            logger.info("Epoch: {}".format(current_epoch))

            for step, (feature_batch, label_batch) in enumerate(
                    feature_label_dataloader
            ):
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
                predictions = self.model(feature_batch)

                loss = self.trainer_config.criterion(predictions, label_batch)

                acc = accuracy_of_probs(predictions, label_batch)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(validation_dataloader), epoch_acc / len(
            validation_dataloader
        )

    def _denoise_rule_matches(self, rule_matches_z: np.ndarray) -> np.ndarray:
        """
        Denoises the applied weak supervision source.
        Args:
            rule_matches_z: Matrix with all applied weak supervision sources. Shape: (Instances x Rules)
        Returns: Denoised / Improved applied labeling function matrix. Shape: (Instances x Rules)
        """
        logger.info("Start denoising labeling functions with k: {}.".format(self.k))

        if self.k == 1:
            return rule_matches_z

        logger.info("This can take a while ...")

        neighbors = NearestNeighbors(n_neighbors=self.k, n_jobs=-1).fit(
            self.tfidf_values
        )
        distances, indices = neighbors.kneighbors(self.tfidf_values)
        new_lfs = activate_all_neighbors(rule_matches_z, indices)
        return new_lfs

    def get_or_create_z(self, prefix: str, k: int) -> np.ndarray:
        path_for_cache = "data/cached_knn/{}_{}_{}".format(prefix, str(k), ".npy")

        if os.path.exists(path_for_cache):
            denoised_rule_matches_z = np.load(path_for_cache, allow_pickle=True)
        else:
            denoised_rule_matches_z = self._denoise_rule_matches(self.rule_matches_z)
            self.cache_matches(path_for_cache, denoised_rule_matches_z)

        return denoised_rule_matches_z

    def cache_matches(
            self, file_path: str, denoised_rule_matches_z: np.ndarray
    ) -> None:
        os.makedirs("data/cached_knn/", exist_ok=True)
        np.save(file_path, denoised_rule_matches_z)

    def print_step_update(self, step: int, max_steps: int):
        if step % 40 == 0 and not step == 0:
            logger.info("  Batch {:>5,}  of  {:>5,}.".format(step, max_steps))
