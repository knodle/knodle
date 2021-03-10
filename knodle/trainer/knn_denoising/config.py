from knodle.trainer.config import MajorityConfig


class KNNConfig(MajorityConfig):
    def __init__(
            self,
            k: int = None,
            radius: float = None,
            weighted_knn_activation: bool = False,
            use_approximation: bool = False,
            activate_no_match_instances: bool = True, 
            caching_folder: str = None,  # if set to string, denoised data is cached
            **kwargs
    ):
        super().__init__(**kwargs)
        self.k = k
        self.radius = radius
        self.weighted_knn_activation = weighted_knn_activation
        self.use_approximation = use_approximation
        self.activate_no_match_instances = activate_no_match_instances
        self.caching_folder = caching_folder

        if self.k is not None and self.radius is not None:
            raise RuntimeError(
                "The Knn trainer can either use the radius or the number of "
                "neighbours to denoise by neighborhood activation"
            )

        if self.k is None and self.use_approximation:
            raise RuntimeError(
                "The Knn trainer can only use the radius for exact neighbor search "
                "Distance-based selection is currently unavailable for approximate NN."
            )
