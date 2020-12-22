import torch
from torch import nn


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int, output_classes: int):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_classes)

    def forward(self, batch):
        """
        Foward method of the model
        Args:
            batch: This is the model input. Every model input will be parsed into a DataLoader and fed to
                the model batch-wise. We extract the first index (0) from the batch to use as features. This varies
                from model to model and is user defined.

        Returns:

        """
        x = batch[0]
        outputs = self.linear(x)
        return outputs
