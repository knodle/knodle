import os

from knodle.trainer.baseline.config import MajorityConfig
from knodle.trainer.knn_denoising.config import KNNConfig

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

    def get_snorkel_cache_file(self):
        file_tags = f"{self.label_model_num_epochs}_{self.filter_non_labelled}"
        if self.caching_suffix:
            file_tags += f"_{self.caching_suffix}"

        cache_file = os.path.join(
            self.caching_folder,
            f"snorkel_denoised_rule_matches_z_{file_tags}.lib"
        )

        return cache_file

@AutoConfig.register("snorkel_knn")
class SnorkelKNNConfig(SnorkelConfig, KNNConfig):
    """Config class for the SnorkelKNNTrainer, which combines k-NN denoising and the Snorkel Label model, in that order.
    See base trainers for configuration possibilities.
    """

    def __init__(self, **kwargs):
        # use all config parameters in SnorkelConfig and KNNConfig
        super().__init__(**kwargs)

    def get_snorkel_cache_file(self):
        nn_type = "ann" if self.use_approximation else "knn"
        file_tags = f"{self.label_model_num_epochs}_{self.filter_non_labelled}_{self.k}_{nn_type}"

        if self.caching_suffix:
            file_tags += f"_{self.caching_suffix}"

        cache_file = os.path.join(
            self.caching_folder,
            f"snorkel_knn_denoised_rule_matches_z_{file_tags}.lib"
        )

        return cache_file