import os
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
            self, patience=4, verbose: bool = False, delta: int = 0, save_model_path: str = None,
            save_model_name: str = None
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        if save_model_path:
            self.save_model_path = save_model_path
        else:
            self.save_model_path = "trained_models"

        if save_model_name:
            self.save_model_name = save_model_name + "_best.pt"
        else:
            self.save_model_name = "checkpoint_best.pt"

    def __call__(self, val_loss, model):

        if self.best_score is not None and val_loss >= self.best_score + self.delta:
            if self.counter < self.patience:
                logger.info(
                    f"Curr val loss: {val_loss}, best is: {self.best_score}. EarlyStopping counter: {self.counter} out "
                    f"of {self.patience}"
                )
                self.counter += 1
            else:
                self.early_stop = True

        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        logger.info(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")

        os.makedirs(self.save_model_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.save_model_path, self.save_model_name))
        logger.info(f"Model saved to {self.save_model_name}")
        self.val_loss_min = val_loss
