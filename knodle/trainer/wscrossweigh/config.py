from torch.optim import Optimizer

from knodle.trainer.baseline.config import MajorityConfig
from knodle.trainer.auto_config import AutoConfig


@AutoConfig.register("wscrossweigh")
class WSCrossWeighDenoisingConfig(MajorityConfig):
    def __init__(
            self,
            partitions: int = 2,
            folds: int = 10,
            weight_reducing_rate: float = 0.5,
            samples_start_weights: float = 2.0,
            cw_epochs: int = None,
            cw_batch_size: int = None,
            cw_optimizer: Optimizer = None,
            cw_lr: int = 0.1,
            cw_filter_non_labelled: bool = None,
            cw_other_class_id: int = None,
            cw_grad_clipping: int = None,
            cw_seed: int = None,
            **kwargs
    ):

        super().__init__(**kwargs)
        self.partitions = partitions
        self.folds = folds
        self.weight_reducing_rate = weight_reducing_rate
        self.samples_start_weights = samples_start_weights

        if cw_grad_clipping is None:
            self.cw_grad_clipping = self.grad_clipping
        else:
            self.cw_grad_clipping = cw_grad_clipping

        if cw_epochs is None:
            self.cw_epochs = self.epochs
        else:
            self.cw_epochs = cw_epochs

        if cw_batch_size is None:
            self.cw_batch_size = self.batch_size
        else:
            self.cw_batch_size = cw_batch_size

        if cw_optimizer is None:
            self.cw_optimizer = self.optimizer
        else:
            self.cw_optimizer = cw_optimizer

        if cw_filter_non_labelled is None and cw_other_class_id is None:
            self.cw_filter_non_labelled = self.filter_non_labelled
            self.cw_other_class_id = self.other_class_id
        else:
            self.cw_filter_non_labelled = cw_filter_non_labelled
            self.cw_other_class_id = cw_other_class_id

        self.cw_seed = cw_seed
        self.cw_lr = cw_lr

