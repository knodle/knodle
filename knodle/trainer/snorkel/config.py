from knodle.trainer.baseline.config import MajorityConfig
from knodle.trainer.knn_denoising.config import KNNConfig

from knodle.trainer.auto_config import AutoConfig


@AutoConfig.register("snorkel")
class SnorkelConfig(MajorityConfig):
    def __init__(
            self,
            label_model_num_epochs: int = 100,
            label_model_log_freq: int = 10,
            caching_folder: str = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.label_model_num_epochs = label_model_num_epochs
        self.label_model_log_freq = label_model_log_freq
        self.caching_folder = caching_folder


@AutoConfig.register("snorkel_knn")
class SnorkelKNNConfig(SnorkelConfig, KNNConfig):
    def __init__(self, **kwargs):
        # use all config parameters in SnorkelConfig and KNNConfig
        super().__init__(**kwargs)
