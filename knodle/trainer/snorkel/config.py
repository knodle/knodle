from knodle.trainer.baseline.config import MajorityConfig
from knodle.trainer.knn_denoising.config import KNNConfig

from knodle.trainer.auto_config import AutoConfig


@AutoConfig.register("snorkel")
class SnorkelConfig(MajorityConfig):
    def __init__(
            self,
            label_model_num_epochs: int = 5000,
            label_model_log_freq: int = 500,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.label_model_num_epochs = label_model_num_epochs
        self.label_model_log_freq = label_model_log_freq


@AutoConfig.register("snorkel_knn")
class SnorkelKNNConfig(SnorkelConfig, KNNConfig):
    def __init__(self, **kwargs):
        # use all config parameters in SnorkelConfig and KNNConfig
        super().__init__(**kwargs)
