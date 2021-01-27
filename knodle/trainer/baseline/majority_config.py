from knodle.trainer.config import TrainerConfig


class MajorityConfig(TrainerConfig):
    def __init__(self, filter_non_labelled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.filter_non_labelled = filter_non_labelled
