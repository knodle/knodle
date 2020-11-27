import logging
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from torch import Tensor
from torch.nn import Module
import numpy as np
from tqdm import tqdm
from knodle.final_label_decider.FinalLabelDecider import get_majority_vote_probabilities
from knodle.trainer.config.TrainerConfig import TrainerConfig
from knodle.trainer.ds_model_trainer.ds_model_trainer import DsModelTrainer
from knodle.trainer.utils import log_section
from knodle.trainer.utils.utils import accuracy_of_probs
from knodle.utils.caching import cache_data

logger = logging.getLogger(__name__)


class KnnDenoising(DsModelTrainer):
    def __init__(self, model: Module, trainer_config: TrainerConfig = None):
        super().__init__(model, trainer_config)

    def train(
        self,
        inputs: Tensor,
        rule_matches: np.ndarray,
        epochs,
        tfidf_values: csr_matrix,
        k: int,
        cache_knn: bool = False,
        cache_path: str = None,
    ):
        """
        This function first denoises the labeling functions with kNN, gets final labels with majoriyt vote and trains
        finally the model.
        Args:
            inputs: Input tensors. These tensors will be fed to the provided model. Shape (n_instaces x n_features)
            rule_matches: Array, shape (n_instances x n_rules)
            tfidf_values: Array, shape (n_instances x n_tfidf_features)
            epochs: Epochs to train
            k: Number of neighbors to consider in kNN approach.
            cache_knn: Should the indices and be cached?
            cache_path: If it should be cached, path to cache
        """

        if len(inputs) != len(rule_matches) != len(tfidf_values):
            # TODO: This can vary if inputs is a more dimensional tensor (see BERT)
            raise ValueError(
                "inputs, rule_matches and tfidf_values need to have the same length"
            )

        if epochs <= 0:
            raise ValueError("Epochs needs to be positive")

        if k <= 0:
            raise ValueError("k needs to be positive")

        if cache_knn and cache_path is None:
            raise AttributeError(
                "If the data should be cached the argument cache_path is required"
            )

        denoised_applied_lfs = self.denoise_rule_matches(
            rule_matches, tfidf_values=tfidf_values, k=k
        )
        labels = get_majority_vote_probabilities(
            rule_matches=denoised_applied_lfs,
            output_classes=self.trainer_config.output_classes,
        )
        labels = Tensor(labels)

        self.model.train()

        log_section("Training starts", logger)

        for current_epoch in range(epochs):
            logger.info("Epoch: {}".format(current_epoch))
            self.model.zero_grad()
            predictions = self.model(inputs)
            loss = self.trainer_config.criterion(predictions, labels)
            logger.info("Loss is: {}".format(loss))
            loss.backward()
            self.trainer_config.optimizer.step()

        log_section("Training done", logger)

    def denoise_rule_matches(
        self,
        rule_matches: np.ndarray,
        tfidf_values: csr_matrix,
        k: int,
        cache_knn: bool = False,
        cache_path: str = None,
    ) -> np.ndarray:
        """
        Denoises the applied weak supervision source.
        Args:
            rule_matches: Matrix with all applied weak supervision sources. Shape: (Instances x Rules)
            tfidf_values: Text in tfidf values. Shape: (Instances x Features)
            k: How many neighbors to consider. Hyperparameter
            cache_knn: Should the indices and be cached?
            cache_path: If it should be cached, path to cache

        Returns: Denoised / Improved applied labeling function matrix. Shape: (Instances x Rules)

        """
        logger.info("Start denoising labeling functions with k: {}.".format(k))

        if k == 1:
            return rule_matches

        logger.info("This can take a while ...")

        neighbors = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(tfidf_values)
        distances, indices = neighbors.kneighbors(tfidf_values)

        if cache_knn:
            try:
                cache_data(indices, cache_path)
            except Exception as err:
                logger.warning(
                    "Couldn't cache data because of following error: {}".format(err)
                )

        new_lfs = self.activate_all_neighbors(rule_matches, indices)
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
                matched_lfs = np.where(lf != -1)[0]
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
