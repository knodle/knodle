from torch.nn import Module
from torch import Tensor
import numpy as np

from knodle.final_label_decider.FinalLabelDecider import get_majority_vote_probabilities
from knodle.model import LogisticRegressionModel
from knodle.trainer.model_config.ModelConfig import ModelConfig


class SimpleDsModelTrainer:
    def __init__(self, model: Module, model_config: ModelConfig = None):
        self.model = model
        if model_config is None:
            self.model_config = ModelConfig(self.model)
        else:
            self.model_config = model_config

    def train(
        self, inputs: Tensor, applied_labeling_functions: np.ndarray, epochs: int
    ):
        self.model.train()
        labels = get_majority_vote_probabilities(
            applied_lfs=applied_labeling_functions,
            output_classes=self.model_config.output_classes,
        )

        labels = Tensor(labels)

        for current_epoch in range(epochs):
            print("Epoch: ", current_epoch)
            self.model.zero_grad()
            predictions = self.model(inputs)
            loss = self.model_config.criterion(predictions, labels)
            print("Loss is: ", loss.float())
            loss.backward()
            self.model_config.optimizer.step()


if __name__ == "__main__":
    logistic_regression = LogisticRegressionModel(10, 2)
    obj = SimpleDsModelTrainer(logistic_regression)
