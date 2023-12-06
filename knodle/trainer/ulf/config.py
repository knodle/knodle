from knodle.trainer.auto_config import AutoConfig
from knodle.trainer.cleanlab.config import CleanLabConfig


@AutoConfig.register("ulf")
class UlfConfig(CleanLabConfig):
    def __init__(
            self,
            use_prior: bool = False,
            p: float = 0.5,         # multiplier of the newly learned t matrix (see update_t_matrix function)
            other_coeff: float = 0.5,
            target: str = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.use_prior = use_prior

        if not self.use_prior:
            self.p = p

        self.other_coeff = other_coeff
        self.target = target
