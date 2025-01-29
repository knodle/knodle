from typing import Optional, Dict

import torch
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class KnodleLightningModule(pl.LightningModule):
    def __init__(
            self,
            train_metrics=None,
            test_metrics: Dict = {"accuracy": torchmetrics.Accuracy()},
            **kwargs
    ):
        super().__init__(**kwargs)

        self.set_metrics("train", train_metrics)
        self.set_metrics("test", test_metrics)

    def set_metrics(self, split: str, metrics: Dict = dict()):
        metrics = {} if metrics is None else metrics

        self.__setattr__(f"{split}_metric_names", metrics.keys())
        for name, metric in metrics.items():
            self.__setattr__(f"{split}_{name}", metric)

    def log_metrics(self, split: str, logits: torch.Tensor, truth: torch.Tensor):
        for name in self.__getattribute__(f"{split}_metric_names"):
            metric = self.__getattr__(f"{split}_{name}")
            metric(logits, truth)
            self.log(f"{split}_{name}", metric)

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        raise NotImplementedError("Please implement training_step")

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        raise NotImplementedError("Please implement validation_step")
