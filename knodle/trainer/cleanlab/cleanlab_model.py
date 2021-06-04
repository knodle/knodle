from torch import nn


class MyModel(nn.Module):
    def forward(self, data, sample_weight):
        """
        Work to do by this model
        """
        # some specific model-dependent work
        return