import os
import logging

import joblib
import numpy as np

from torch.optim import SGD
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from knodle.transformation.majority import input_to_majority_vote_input
from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.trainer import Trainer
from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.baseline.no_denoising import NoDenoisingTrainer
from knodle.trainer.knn_denoising.config import KNNConfig
from knodle.trainer.utils.denoise import activate_neighbors

logger = logging.getLogger(__name__)


@AutoTrainer.register('knn_trainer')
class KnnDenoisingTrainer(NoDenoisingTrainer):
    def __init__(
            self,
            knn_feature_matrix: np.ndarray = None,
            **kwargs
    ):
        if kwargs.get("trainer_config") is None:
            kwargs["trainer_config"] = KNNConfig(optimizer=SGD(kwargs.get("model").parameters(), lr=0.001))
        super().__init__(**kwargs)

        if knn_feature_matrix is None:
            self.knn_feature_matrix = csr_matrix(self.model_input_x.tensors[0].numpy())
        else:
            self.knn_feature_matrix = knn_feature_matrix

    def train(self):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """

        denoised_rule_matches_z = self._knn_denoise_rule_matches()

        model_input_x, label_probs = input_to_majority_vote_input(
            self.model_input_x, denoised_rule_matches_z, self.mapping_rules_labels_t,
            filter_non_labelled=self.trainer_config.filter_non_labelled
        )

        feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self.train_loop(feature_label_dataloader)

    def _knn_denoise_rule_matches(self) -> np.ndarray:
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
            neighbors = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(self.knn_feature_matrix)
            distances, indices = neighbors.kneighbors(self.knn_feature_matrix, n_neighbors=k)
        else:
            neighbors = NearestNeighbors(radius=self.trainer_config.radius, n_jobs=-1).fit(self.knn_feature_matrix)
            distances, indices = neighbors.radius_neighbors(self.knn_feature_matrix)

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
