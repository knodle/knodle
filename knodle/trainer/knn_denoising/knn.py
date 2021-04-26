import os
import logging

import joblib
import numpy as np

from torch.optim import SGD
from torch.utils.data import TensorDataset

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex

from knodle.transformation.majority import input_to_majority_vote_input
from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.baseline.majority import MajorityVoteTrainer
from knodle.trainer.knn_denoising.config import KNNConfig
from knodle.trainer.utils.denoise import activate_neighbors

logger = logging.getLogger(__name__)


@AutoTrainer.register('knn')
class KnnDenoisingTrainer(MajorityVoteTrainer):
    def __init__(
            self,
            knn_feature_matrix: np.ndarray = None,
            **kwargs
    ):
        if kwargs.get("trainer_config") is None:
            kwargs["trainer_config"] = KNNConfig(optimizer=SGD, lr=0.001)
        super().__init__(**kwargs)

        if knn_feature_matrix is None:
            self.knn_feature_matrix = csr_matrix(self.model_input_x.tensors[0].numpy())
        else:
            self.knn_feature_matrix = knn_feature_matrix

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ):
        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)

        # initialise optimizer
        self.trainer_config.optimizer = self.initialise_optimizer()

        self.rule_matches_z = self.rule_matches_z.astype(np.int8)
        self.mapping_rules_labels_t = self.mapping_rules_labels_t.astype(np.int8)

        self._knn_denoise_rule_matches()

        model_input_x, label_probs = input_to_majority_vote_input(
            self.model_input_x, self.rule_matches_z, self.mapping_rules_labels_t.astype(np.int64),
            filter_non_labelled=self.trainer_config.filter_non_labelled,
            other_class_id=self.trainer_config.other_class_id
        )

        feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self._train_loop(feature_label_dataloader)

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
        if self.trainer_config.caching_folder:
            cache_file = self.trainer_config.get_cache_file()
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
                knn_matrix_shape = self.knn_feature_matrix.shape

                logger.info("Creating annoy index...")
                t = AnnoyIndex(knn_matrix_shape[1], 'dot')
                for i, v in enumerate(self.knn_feature_matrix):
                    if not ignore[i]:
                        t.add_item(i, v)

                t.build(10, n_jobs=self.trainer_config.n_jobs)

                self.knn_feature_matrix = None

                logger.info("Retrieving neighbor indices...")
                indices = (         # make a generator: no memory is allocated at this moment
                    np.array(t.get_nns_by_item(i, k, search_k=-1, include_distances=False))
                    if not ignore[i] else np.array([])
                    for i in range(knn_matrix_shape[0])
                )
            else:
                # possible radius implementation; delete error in config then
                pass
        else:
            # use standard precise kNN
            if k is not None:
                logger.info("Creating NN index...")
                neighbors = NearestNeighbors(n_neighbors=k, n_jobs=self.trainer_config.n_jobs)\
                    .fit(self.knn_feature_matrix)
                logger.info("Retrieving neighbor indices...")
                indices = neighbors.kneighbors(self.knn_feature_matrix, n_neighbors=k, return_distance=False)
            else:
                logger.info("Creating NN index...")
                neighbors = NearestNeighbors(radius=self.trainer_config.radius, n_jobs=self.trainer_config.n_jobs)\
                    .fit(self.knn_feature_matrix)
                logger.info("Retrieving neighbor indices...")
                indices = neighbors.radius_neighbors(self.knn_feature_matrix, return_distance=False)

        # activate matches.
        logger.info("Activating neighbors...")
        self.rule_matches_z = activate_neighbors(self.rule_matches_z, indices)

        # save data for caching
        if self.trainer_config.caching_folder:
            os.makedirs(self.trainer_config.caching_folder, exist_ok=True)
            joblib.dump(self.rule_matches_z, cache_file)

        return self.rule_matches_z

    def print_step_update(self, step: int, max_steps: int):
        if step % 40 == 0 and not step == 0:
            logger.info(f"  Batch {step}  of  {max_steps}.")
