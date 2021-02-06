from knodle.trainer.config import TrainerConfig


class MajorityConfig(TrainerConfig):
    def __init__(self,
                 filter_emtpy_z_rows: bool = True,
                 use_probabilistic_labels: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.filter_emtpy_z_rows = filter_emtpy_z_rows
        self.use_probabilistic_labels = use_probabilistic_labels
