from knodle.trainer.baseline.config import MajorityConfig

from knodle.trainer.auto_config import AutoConfig


@AutoConfig.register("cleanlab")
class CleanLabConfig(MajorityConfig):
    def __init__(
            self,
            cv_n_folds=5,
            prune_method='prune_by_noise_rate',
            converge_latent_estimates=False,
            pulearning=None,
            n_jobs=None,
            psx_with_dscrossweigh=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.cv_n_folds = cv_n_folds
        self.prune_method = prune_method
        self.converge_latent_estimates = converge_latent_estimates
        self.pulearning = pulearning
        self.n_jobs = n_jobs
        self.psx_with_dscrossweigh = psx_with_dscrossweigh
