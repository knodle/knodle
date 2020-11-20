import logging
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import Union

from torch import Tensor
from torch.nn import Module
import numpy as np
from tqdm import tqdm
from knodle.final_label_decider.FinalLabelDecider import get_majority_vote_probabilities
from knodle.trainer.model_config.ModelConfig import ModelConfig

logger = logging.getLogger(__name__)


class KnnDenoising:
    def __init__(self, model: Module, model_config: ModelConfig = None):
        self.model = model
        if model_config is None:
            self.model_config = ModelConfig(self.model)
            logger.info("Default Model Config is used: {}".format(self.model_config))
        else:
            self.model_config = model_config
            logger.info(
                "Initalized trainer with custom model config: {}".format(
                    self.model_config.__dict__
                )
            )

    def train(
        self,
        inputs: Tensor,
        applied_labeling_functions: np.ndarray,
        tfidf_values: Union[csr_matrix, np.ndarray],
        epochs: int,
        k: int,
    ):
        denoised_applied_lfs = self.denoise_applied_lfs(
            applied_labeling_functions, tfidf_values, k=k
        )
        labels = get_majority_vote_probabilities(
            applied_lfs=denoised_applied_lfs,
            output_classes=self.model_config.output_classes,
        )
        labels = Tensor(labels)

        self.model.train()

        for current_epoch in range(epochs):
            logger.info("Epoch: {}".format(current_epoch))
            self.model.zero_grad()
            predictions = self.model(inputs)
            loss = self.model_config.criterion(predictions, labels)
            logger.info("Loss is: {}".format(loss))
            loss.backward()
            self.model_config.optimizer.step()

    def denoise_applied_lfs(
        self,
        applied_labeling_functions: np.ndarray,
        tfidf_values: csr_matrix,
        k: int = 1,
    ) -> np.ndarray:
        """
        Denoises the applied weak supervision source.
        Args:
            applied_labeling_functions: Matrix with all applied weak supervision sources. Shape: (Instances x Rules)
            tfidf_values: Text in tfidf values. Shape: (Instances x Features)
            k: How many neighbors to consider. Hyperparameter

        Returns: Denoised / Improved applied labeling function matrix. Shape: (Instances x Rules)

        """
        logger.info("Start denoising labeling functions with k: {}.".format(k))
        if k == 1:
            return applied_labeling_functions

        logger.info("This can take a while ...")

        neighbors = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(tfidf_values)
        distances, indices = neighbors.kneighbors(tfidf_values)
        new_lfs = self.activate_all_neighbors(applied_labeling_functions, indices)
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
