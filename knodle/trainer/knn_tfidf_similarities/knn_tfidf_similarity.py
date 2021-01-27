import logging

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from knodle.trainer import TrainerConfig
from knodle.trainer.trainer import Trainer
from knodle.trainer.utils import log_section
from knodle.trainer.utils.denoise import get_majority_vote_probs
from knodle.trainer.utils.filter import filter_empty_probabilities
from knodle.trainer.utils.utils import accuracy_of_probs, extract_tensor_from_dataset

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
        trainer_config: TrainerConfig = None,
    ):
        super().__init__(
            model, mapping_rules_labels_t, model_input_x, rule_matches_z, trainer_config
        )
        self.tfidf_values = tfidf_values
        self.k = k

    def train(self):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """

        denoised_rule_matches_z = self._denoise_rule_matches(self.rule_matches_z)

        label_probs = get_majority_vote_probs(
            denoised_rule_matches_z, self.mapping_rules_labels_t
        )

        model_input_x, label_probs = filter_empty_probabilities(self.model_input_x, label_probs)

        model_input_x_tensor = extract_tensor_from_dataset(model_input_x, 0)
        feature_label_dataset = TensorDataset(model_input_x_tensor, Tensor(label_probs))
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        log_section("Training starts", logger)

        self.model.train()
        for current_epoch in tqdm(range(self.trainer_config.epochs)):
            epoch_loss, epoch_acc = 0.0, 0.0
            logger.info("Epoch: {}".format(current_epoch))

            for feature_batch, label_batch in feature_label_dataloader:
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

            logger.info("Epoch loss: {}".format(avg_loss))
            logger.info("Epoch Accuracy: {}".format(avg_acc))

        log_section("Training done", logger)

    def _make_dataloader(self, dataset: TensorDataset) -> DataLoader:
        dataloader = DataLoader(
            dataset, batch_size=self.trainer_config.batch_size, drop_last=False
        )
        return dataloader

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
        new_lfs = self._activate_all_neighbors(rule_matches_z, indices)
        return new_lfs

    def _activate_all_neighbors(
        self, lfs: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        """
        Find all closest neighbors and take the same label ids
        Args:
            lfs:
            indices:
        Returns:
        """
        new_lfs_array = np.full(lfs.shape, fill_value=-1)

        for index, lf in tqdm(enumerate(lfs)):

            try:
                matched_lfs = np.where(lf != 0)[0]
                if len(matched_lfs) == 0:
                    continue
                matched_lfs = matched_lfs[:, np.newaxis]
                neighbors = indices[index]
                to_replace = new_lfs_array[neighbors, matched_lfs]
                label_matched_lfs = lf[matched_lfs][:, 0]
                tiled_labels = np.tile(
                    np.array(label_matched_lfs), (to_replace.shape[1], 1)
                ).transpose()
                new_lfs_array[neighbors, matched_lfs] = tiled_labels
            except IndexError:
                pass

        return new_lfs_array
