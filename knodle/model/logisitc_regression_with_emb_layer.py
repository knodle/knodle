import torch
from torch import nn
import numpy as np


class LogisticRegressionModel(nn.Module):
    def __init__(
            self,
            input_size: int,
            word_input_dim: int,
            word_output_dim: int,
            word_embedding_matrix: np.ndarray,
            output_classes: int,
    ):
        super(LogisticRegressionModel, self).__init__()

        self.word_embedding = nn.Embedding(
            word_input_dim, word_output_dim, padding_idx=0
        )
        self.word_embedding.weight = nn.Parameter(
            torch.tensor(word_embedding_matrix, dtype=torch.float32)
        )
        self.word_embedding.weight.requires_grad = False

        # self.td_dense = nn.Linear(input_size * word_output_dim, size_factor)
        self.linear = nn.Linear(input_size * word_output_dim, output_classes)

    def forward(self, x):
        word_embeddings = self.word_embedding(x)
        word_embeddings = word_embeddings.view(x.shape[0], -1)
        outputs = self.linear(word_embeddings)
        return outputs
