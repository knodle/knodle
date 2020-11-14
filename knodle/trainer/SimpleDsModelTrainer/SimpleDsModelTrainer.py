from torch.nn import Module
from torch import Tensor
import numpy as np

from knodle.final_label_decider.FinalLabelDecider import get_majority_vote_probabilities
from knodle.model import LogisticRegressionModel
from knodle.trainer.utils.utils import create_criterion, create_optimizer


class SimpleDsModelTrainer():

    def __init__(self, model: Module, criterion: str = 'cross_entropy_with_probs', optimizer='SGD',
                 output_classes: int = 2):
        self.model = model
        self.criterion = create_criterion(criterion)
        self.optimizer = create_optimizer(self.model, optimizer)
        self.output_classes = output_classes

    def train(self, inputs: Tensor, applied_labeling_functions: np.ndarray, epochs: int):
        self.model.train()
        labels = get_majority_vote_probabilities(applied_lfs=applied_labeling_functions,
                                                 output_classes=self.output_classes)

        labels = Tensor(labels)

        for current_epoch in range(epochs):
            print("Epoch: ", current_epoch)
            self.model.zero_grad()
            predictions = self.model(inputs)
            loss = self.criterion(predictions, labels)
            print("Loss is: ", loss.float())
            loss.backward()
            self.optimizer.step()


if __name__ == '__main__':
    logistic_regression = LogisticRegressionModel(10, 2)
    obj = SimpleDsModelTrainer(logistic_regression)
