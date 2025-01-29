from typing import Optional

from pytorch_lightning import Trainer, LightningModule, LightningDataModule


class KnodleLightningLabelModelTrainer(Trainer):
    def __init__(
            self,
            use_probabilistic_labels: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.use_probabilistic_labels = use_probabilistic_labels
        self.probability_threshold = None

    def apply_label_model(self, datamodule):
        raise NotImplementedError("Please implement")

    def fit(
            self,
            model: LightningModule,
            datamodule: Optional[LightningDataModule] = None,
            **kwargs
    ):
        datamodule = self.apply_label_model(datamodule)
        model.total_steps = len(datamodule.train_dataloader())

        super().fit(model=model, datamodule=datamodule)
