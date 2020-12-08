import logging
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from knodle.trainer import TrainerConfig
from knodle.trainer.ds_model_trainer.ds_model_trainer import DsModelTrainer
from knodle.trainer.utils import log_section
from knodle.trainer.utils.utils import accuracy_of_probs

logger = logging.getLogger(__name__)


class KnnTfidfSimilarity(DsModelTrainer):
    def __init__(
        self,
        model: Module,
        mapping_rules_labels_t: np.ndarray,
        tfidf_values: csr_matrix,
        k: int,
        trainer_config: TrainerConfig = None,
    ):
        super().__init__(model, mapping_rules_labels_t, trainer_config)
        self.tfidf_values = tfidf_values
        self.k = k

    def train(
        self,
        model_input_x: TensorDataset,
        rule_matches_z: np.ndarray,
        epochs: int,
    ):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        Args:
            model_input_x: Input tensors. These tensors will be fed to the provided model.
            rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
            epochs: Epochs to train
        """

        if epochs <= 0:
            raise ValueError("Epochs needs to be positive")

        denoised_rule_matches_z = self.denoise_rule_matches(rule_matches_z)

        labels = self.get_majority_vote_probs(denoised_rule_matches_z)

        label_dataset = TensorDataset(Tensor(labels))

        feature_dataloader = self.make_dataloader(model_input_x)
        label_dataloader = self.make_dataloader(label_dataset)
        log_section("Training starts", logger)

        self.model.train()
        for current_epoch in tqdm(range(epochs)):
            epoch_loss, epoch_acc = 0.0, 0.0
            logger.info("Epoch: {}".format(current_epoch))

            for feature_batch, label_batch in zip(feature_dataloader, label_dataloader):
                labels = label_batch[0]
                self.model.zero_grad()
                predictions = self.model(feature_batch)
                loss = self.trainer_config.criterion(predictions, labels)
                loss.backward()
                self.trainer_config.optimizer.step()
                acc = accuracy_of_probs(predictions, labels)

                epoch_loss += loss.detach()
                epoch_acc += acc.item()

            avg_loss = epoch_loss / len(feature_dataloader)
            avg_acc = epoch_acc / len(feature_dataloader)

            logger.info("Epoch loss: {}".format(avg_loss))
            logger.info("Epoch Accuracy: {}".format(avg_acc))

        log_section("Training done", logger)

    def make_dataloader(self, dataset: TensorDataset) -> DataLoader:
        dataloader = DataLoader(
            dataset, batch_size=self.trainer_config.batch_size, drop_last=True
        )
        return dataloader

    def get_majority_vote_probs(self, rule_matches_z: np.ndarray):
        """
        This function calculates a majority vote probability for all rule_matches_z. First rule counts will be
        calculated,
        then a probability will be calculated by dividing the values row-wise with the sum. To counteract zero
        division
        all nan values are set to zero.
        Args:
            rule_matches_z: Binary encoded array of which rules matched. Shape: instances x rules
        Returns:

        """
        rule_counts = np.matmul(rule_matches_z, self.mapping_rules_labels_t)
        rule_counts_probs = rule_counts / rule_counts.sum(axis=1).reshape(-1, 1)

        rule_counts_probs[np.isnan(rule_counts_probs)] = 0
        return rule_counts_probs

    def denoise_rule_matches(self, rule_matches_z: np.ndarray) -> np.ndarray:
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
        new_lfs = self.activate_all_neighbors(rule_matches_z, indices)
        return new_lfs

    def activate_all_neighbors(
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
                # IndexError can occur in some special cases
                pass

        return new_lfs_array

    def test(self, test_features: Tensor, test_labels: Tensor):
        self.model.eval()

        predictions = self.model(test_features)

        acc = accuracy_of_probs(predictions, test_labels)
        logger.info("Accuracy is {}".format(acc.detach()))
        return acc
