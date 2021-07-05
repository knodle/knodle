from knodle.trainer.baseline.config import MajorityConfig
from knodle.trainer.knn_aggregation.config import KNNConfig

from knodle.trainer.auto_config import AutoConfig


@AutoConfig.register("snorkel")
class SnorkelConfig(MajorityConfig):
    """Config class for the Snorkel wrapper.
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


@AutoConfig.register("snorkel_knn")
class SnorkelKNNConfig(SnorkelConfig, KNNConfig):
    """Config class for the SnorkelkNNTrainer, which combines k-NN denoising and the Snorkel Label model, in that order.
    See base trainers for configuration possibilities.
    """

    def __init__(self, **kwargs):
        # use all config parameters in SnorkelConfig and KNNConfig
        super().__init__(**kwargs)
