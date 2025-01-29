import torch
import torch.nn as nn
import torch.nn.functional as F

from knodle.trainer.lightning.base_model import KnodleLightningModule


class LitModel(KnodleLightningModule):
    def __init__(
            self,
            max_features=4000,
            num_classes=2,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.l1 = nn.Linear(max_features, max_features // 2)
        self.l2 = nn.Linear(max_features // 2, max_features // 4)
        self.l3 = nn.Linear(max_features, num_classes)

    def forward(self, x):
        # x = torch.relu(self.l1(x))
        # x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        return F.softmax(x, dim=-1)

    def training_step(self, batch, batch_idx):
        x = batch["x"].float()
        y = batch["labels"].long()

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("loss", loss)
        self.log_metrics("train", y_hat, y)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0].float()
        y = batch[1].long()
        y_hat = self(x)

        self.log_metrics("test", y_hat, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
