import os

from knodle.trainer.baseline.config import MajorityConfig
from knodle.trainer.auto_config import AutoConfig


@AutoConfig.register("knn")
class KNNConfig(MajorityConfig):
    def __init__(
            self,
            k: int = None,
            radius: float = None,
            use_approximation: bool = False,
            activate_no_match_instances: bool = True,
            n_jobs_for_index: int = 4,
            **kwargs
    ):
        """
        A default configuration of KNNConfig Trainer.

        :param k: number of neighbors that are to be found
        :param radius: a radius of a point or points the neighbors are to be found within
        :param use_approximation: if set to True, the approximated ANN will be used instead of kNN
        :param activate_no_match_instances: if set to True, the zero-match rows for knn construction & activation will
        be ignored
        :param n_jobs_for_index:

        """
        super().__init__(**kwargs)
        self.k = k
        self.radius = radius
        self.use_approximation = use_approximation
        self.activate_no_match_instances = activate_no_match_instances
        self.n_jobs = n_jobs_for_index

        if self.k is not None and self.radius is not None:
            raise RuntimeError(
                "The kNN trainer can either use the radius or the number of "
                "neighbours to denoise by neighborhood activation"
            )

        if self.k is None and self.use_approximation:
            raise RuntimeError(
                "The kNN trainer can only use the radius for exact neighbor search "
                "Distance-based selection is currently unavailable for approximate NN."
            )

        # Currently impossible, though can (should?) be done in the future.
        if not self.use_approximation and not self.activate_no_match_instances:
            raise RuntimeError(
                "The kNN trainer with exact neighbor selection always uses all of the instances. "
                "Either 'activate_no_match_instances' or 'use_approximation' has to be set to True."
            )

        # if both number of neighbors and radius size are empty, set k=2 by default
        if self.k is None and self.radius is None:
            self.k = 2

    def get_cache_file(self):
        nn_type = "ann" if self.use_approximation else "knn"
        file_tags = f"{self.k}_{nn_type}"
        if self.caching_suffix:
            file_tags += f"_{self.caching_suffix}"

        cache_file = os.path.join(
            self.caching_folder,
            f"denoised_rule_matches_z_{file_tags}.lib"
        )

        return cache_file
