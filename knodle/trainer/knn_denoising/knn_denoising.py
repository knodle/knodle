import os
import logging

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from torch.nn import Module
from torch.utils.data import TensorDataset

from knodle.transformation.majority import input_to_majority_vote_input
from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.baseline.no_denoising import NoDenoisingTrainer
from knodle.trainer.knn_denoising.knn_config import KNNConfig
from knodle.trainer.utils.denoise import activate_neighbors

logger = logging.getLogger(__name__)


class KnnDenoisingTrainer(NoDenoisingTrainer):
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
            model, mapping_rules_labels_t, model_input_x, rule_matches_z, trainer_config=trainer_config
        )

    def train(self):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """

        denoised_rule_matches_z = self._denoise_rule_matches()

        model_input_x, label_probs = input_to_majority_vote_input(
            self.model_input_x, denoised_rule_matches_z, self.mapping_rules_labels_t,
            filter_non_labelled=self.trainer_config.filter_non_labelled
        )

        feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self.train_loop(feature_label_dataloader)


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
