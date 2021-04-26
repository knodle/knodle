import torch
from torch import nn


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int, output_classes: int):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_classes)

    def forward(self, x):
        x = x.float()
        outputs = self.linear(x)
        return outputs
