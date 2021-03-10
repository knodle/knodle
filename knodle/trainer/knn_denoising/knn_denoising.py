import os
import logging
import gc

import joblib
import numpy as np

from torch.optim import SGD
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex

from knodle.transformation.majority import input_to_majority_vote_input
from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.baseline.no_denoising import NoDenoisingTrainer
from knodle.trainer.knn_denoising.config import KNNConfig
from knodle.trainer.utils.denoise import activate_neighbors

logger = logging.getLogger(__name__)


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
        print(f"denoised_rule: {denoised_rule_matches_z.shape}")
        
        self.rule_matches_z = None
        gc.collect()
        
        print(f"denoised_rule: {denoised_rule_matches_z.shape}")
        print(f"non-zero rows: {(denoised_rule_matches_z.sum(-1) != 0).sum()}")
        
        model_input_x, label_probs = input_to_majority_vote_input(
            self.model_input_x, denoised_rule_matches_z, self.mapping_rules_labels_t.astype(np.int64),
            filter_non_labelled=self.trainer_config.filter_non_labelled,
            other_class_id=self.trainer_config.other_class_id
        )

        print(f"model input: {model_input_x.tensors[0].shape} label_probs: {label_probs.shape}")
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
        k = self.trainer_config.k
        if k == 1:
            return self.rule_matches_z

        # load cached data, if available
        cache_dir = self.trainer_config.caching_folder
        if cache_dir is not None:
            nn_type = "ann" if self.trainer_config.use_approximation else "knn"
            cache_file = os.path.join(cache_dir, f"denoised_rule_matches_z_{k}_{nn_type}.lib")
            if os.path.isfile(cache_file):
                logger.info(f"Loaded knn matrix from cache: {cache_file}")
                return joblib.load(cache_file)

        logger.info(f"Start denoising labeling functions with k: {k}.")
        
        # ignore zero-match rows for knn construction & activation
        if self.trainer_config.activate_no_match_instances:
            ignore = np.zeros((self.knn_feature_matrix.shape[0],), dtype=np.bool)
        else:
            ignore = self.rule_matches_z.sum(-1) == 0

        # Set up data structure, to quickly find nearest neighbors
        if self.trainer_config.use_approximation:
            # use annoy fast ANN
            if k is not None:
                indices = []
                knn_matrix_shape = self.knn_feature_matrix.shape
                
                logger.info("Creating annoy index...")
                t = AnnoyIndex(knn_matrix_shape[1], 'dot')
                for i, v in enumerate(self.knn_feature_matrix):
                    if not ignore[i]:
                        t.add_item(i, v)

                t.build(10, n_jobs=10)
                
                # free RAM
                self.knn_feature_matrix = None
                gc.collect()
                
                logger.info("Retrieving neighbor indices...")
                indices = ( # make a generator: no memory is allocated at this moment
                    np.array(t.get_nns_by_item(i, k, search_k=-1, include_distances=False)) 
                    if not ignore[i] else np.array([])
                    for i in range(knn_matrix_shape[0])
                )
            else:
                pass
        else:
            # use standard precise kNN
            if k is not None:
                logger.info("Creating NN index...")
                neighbors = NearestNeighbors(n_neighbors=k, n_jobs=10).fit(self.knn_feature_matrix)
                logger.info("Retrieving neighbor indices...")
                indices = neighbors.kneighbors(self.knn_feature_matrix, n_neighbors=k, return_distance=False)
            else:
                logger.info("Creating NN index...")
                neighbors = NearestNeighbors(radius=self.trainer_config.radius, n_jobs=10).fit(self.knn_feature_matrix)
                logger.info("Retrieving neighbor indices...")
                indices = neighbors.radius_neighbors(self.knn_feature_matrix, return_distance=False)
                
        # activate matches.
        logger.info("Activating neighbors...")
        denoised_rule_matches_z = activate_neighbors(self.rule_matches_z, indices)

        # save data for caching
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            joblib.dump(denoised_rule_matches_z, cache_file)

        return denoised_rule_matches_z

    def print_step_update(self, step: int, max_steps: int):
        if step % 40 == 0 and not step == 0:
            logger.info(f"  Batch {step}  of  {max_steps}.")

