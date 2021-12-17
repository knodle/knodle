from knodle.trainer.baseline.config import MajorityConfig
from knodle.trainer.auto_config import AutoConfig


@AutoConfig.register("flyingsquid")
class FlyingSquidConfig(MajorityConfig):
    """Config class for the FlyingSquid wrapper.
    """

    def __init__(
            self,
            label_model_num_epochs: int = 5000,
            label_model_log_freq: int = 500,
            **kwargs
    ):
        """
        :param label_model_num_epochs: Number of epochs to train the Label model, which computes P(Y, Z, T).
        :param label_model_log_freq: Logging frequency.
        """
        super().__init__(**kwargs)
        self.label_model_num_epochs = label_model_num_epochs
        self.label_model_log_freq = label_model_log_freq


