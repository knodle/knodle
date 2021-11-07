import logging
from typing import Callable

from torch import Tensor
from torch.optim.optimizer import Optimizer

from knodle.trainer.baseline.config import MajorityConfig
from knodle.trainer.auto_config import AutoConfig


logger = logging.getLogger(__name__)


@AutoConfig.register("cleanlab")
class CleanLabConfig(MajorityConfig):
    def __init__(
            self,
            cv_n_folds=5,
            iterations=1,
            prune_method: str = 'prune_by_noise_rate',
            converge_latent_estimates=False,
            pulearning: int = None,
            py_method: str = 'cnt',
            n_jobs: int = None,
            psx_calculation_method: str = 'random',
            noise_matrix: str = 'rule2class',
            calibrate_cj_matrix: bool = True,
            psx_epochs: int = None,
            psx_optimizer: Optimizer = None,
            psx_lr: float = None,
            psx_criterion: Callable[[Tensor, Tensor], float] = None,
            **kwargs
    ):
        """
        All CleanLab specific parameters (except for psx_calculation_method parameter) are inherited from the original
        CleanLab code (and so are their descriptions).

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

        :param psx_calculation_method: the way how the samples will be splitted into folds for k fold cross validation.
        In the original Cleanlab configuration, it is done randomly. Additionally, we suggest to split data by rules
        matched in them, or by signatures - the sample codification of rules matched in it (i.e., if rules 1, 4 and 5
        matched in a sample, its signature would be "1_4_5").

        :param c_matrix: the C matrix configuration. In the original Cleanlab configuration, it has a shape
        (classes x classes). We add the functionality to calculate the matrix (rules x classes). That is, the confident
        join will be calculated pro rule, but with out-of-sample predicted labels aggregated by classes.
        """

        super().__init__(**kwargs)
        self.cv_n_folds = cv_n_folds
        self.iterations = iterations

        self.prune_method = prune_method
        self.converge_latent_estimates = converge_latent_estimates
        self.pulearning = pulearning
        self.py_method = py_method
        self.n_jobs = n_jobs
        self.psx_calculation_method = psx_calculation_method
        self.noise_matrix = noise_matrix
        self.calibrate_cj_matrix = calibrate_cj_matrix

        if psx_criterion:
            self.psx_criterion = psx_criterion
        else:
            self.psx_criterion = self.criterion

        if psx_epochs:
            self.psx_epochs = psx_epochs
        else:
            self.psx_epochs = self.epochs

        if psx_optimizer:
            self.psx_optimizer = psx_optimizer
        else:
            self.psx_optimizer = self.optimizer

        if psx_lr:
            self.psx_lr = psx_lr
        else:
            self.psx_lr = self.lr

        if self.use_probabilistic_labels:
            logger.warning(
                "WSCleanlab denoising method is not compatible with probabilistic labels. "
                "The labels for each sample will be chosen with majority voting instead."
            )
            self.use_probabilistic_labels = False
