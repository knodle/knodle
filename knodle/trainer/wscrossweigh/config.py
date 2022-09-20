from torch.optim import Optimizer

from knodle.trainer.baseline.config import MajorityConfig
from knodle.trainer.auto_config import AutoConfig


@AutoConfig.register("wscrossweigh")
class WSCrossWeighConfig(MajorityConfig):
    def __init__(
            self,
            partitions: int = 2,
            folds: int = 10,
            weight_reducing_rate: float = 0.5,
            samples_start_weights: float = 2.0,
            cw_epochs: int = None,
            cw_batch_size: int = None,
            cw_optimizer: Optimizer = None,
            cw_lr: int = None,
            cw_unmatched_strategy: str = None,
            cw_ties_strategy: str = None,
            cw_other_class_id: int = None,
            cw_grad_clipping: int = None,
            cw_seed: int = None,
            draw_plot: bool = False,
            **kwargs
    ):
        """
        A default configuration of WSCrossWeigh Trainer.

        :param partitions: number of times the sample weights calculation procedure will be performed
        (with different folds splitting)
        :param folds: number of folds the samples will be splitted into in each sample weights calculation iteration
        :param weight_reducing_rate: a value the sample weight is reduced by each time the sample is misclassified
        :param samples_start_weights: a start weight of all samples
        :param cw_epochs: number of epochs WSCrossWeigh models are to be trained
        :param cw_batch_size: batch size for WSCrossWeigh models training
        :param cw_optimizer: optimizer for WSCrossWeigh models training
        :param cw_lr: learning rate for WSCrossWeigh models training
        :param cw_unmatched_strategy: what to do with samples without matched in WSCrossWeigh
        :param cw_ties_strategy: what to do with samples with no clear majority vote in WSCrossWeigh
        :param cw_other_class_id: id of the negative class; if set, the samples with no rule matched will be assigned to
        it in WSCrossWeigh
        :param cw_grad_clipping: if set to True, gradient norm of an iterable of parameters will be clipped in WSCrossWeigh
        :param cw_seed: the desired seed for generating random numbers in WSCrossWeigh
        :param draw_plot: draw a plot of development data (accuracy & loss)
        """

        super().__init__(**kwargs)
        self.draw_plot = draw_plot
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

        if cw_unmatched_strategy is None:
            self.cw_unmatched_strategy = self.unmatched_strategy
        else:
            self.cw_unmatched_strategy = cw_unmatched_strategy

        if cw_ties_strategy is None:
            self.cw_ties_strategy = self.ties_strategy
        else:
            self.cw_ties_strategy = cw_ties_strategy

        if cw_other_class_id is None:
            self.cw_other_class_id = self.other_class_id
        else:
            self.cw_other_class_id = cw_other_class_id

        if cw_lr is None:
            self.cw_lr = self.lr
        else:
            self.cw_lr = cw_lr

        self.cw_seed = cw_seed

