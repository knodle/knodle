from knodle.trainer.config import MajorityConfig


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

