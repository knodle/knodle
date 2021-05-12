from knodle.trainer.baseline.config import MajorityConfig

from knodle.trainer.auto_config import AutoConfig


@AutoConfig.register("cleanlab")
class CleanLabConfig(MajorityConfig):
    def __init__(
            self,
            cv_n_folds=5,
            prune_method: str = 'prune_by_noise_rate',
            converge_latent_estimates=False,
            pulearning=None,
            n_jobs: int = None,
            psx_calculation_method: str = 'random',
            **kwargs
    ):
        """
        All CleanLab specific parameters are inherited from the original CleanLab code (and so are their descriptions)

        :param cv_n_folds: int
          This class needs holdout predicted probabilities for every data example
          and if not provided, uses cross-validation to compute them.
          cv_n_folds sets the number of cross-validation folds used to compute
          out-of-sample probabilities for each example in X.

        :param prune_method: str (default: 'prune_by_noise_rate')
          Available options: 'prune_by_class', 'prune_by_noise_rate', or 'both'.
          This str determines the method used for pruning.
          1. 'prune_by_noise_rate': works by removing examples with *high probability* of being mislabeled for every
            non-diagonal in the `prune_counts_matrix` (see pruning.py).
          2. 'prune_by_class': works by removing the examples with *smallest probability* of belonging to their given
            class label for every class.
          3. 'both': Finds the examples satisfying (1) AND (2) and removes their set conjunction.

        :param converge_latent_estimates: bool (Default: False)
          If true, forces numerical consistency of latent estimates. Each is
          estimated independently, but they are related mathematically with closed
          form equivalences. This will iteratively enforce consistency.

        :param pulearning : int (0 or 1, default: None).
            Only works for 2 class datasets. Set to the integer of the class that is perfectly labeled
            (certain no errors in that class).

        :param n_jobs: int (Windows users may see a speed-up with n_jobs = 1).
            Number of processing threads used by
            multiprocessing. Default None sets to the number of processing threads on your CPU. Set this to 1 to REMOVE
            parallel processing (if its causing issues).
        """

        super().__init__(**kwargs)
        self.cv_n_folds = cv_n_folds
        self.prune_method = prune_method
        self.converge_latent_estimates = converge_latent_estimates
        self.pulearning = pulearning
        self.n_jobs = n_jobs
        self.psx_calculation_method = psx_calculation_method
