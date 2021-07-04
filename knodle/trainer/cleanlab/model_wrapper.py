from skorch import NeuralNet
from torch import nn

from knodle.trainer import TrainerConfig


class SkorchModel(NeuralNet):
    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
        sample_weight = X['sample_weight']
        return (sample_weight * loss_unreduced).mean()


def wrap_model(model: nn.Module, trainer_config: TrainerConfig):
    """ The function wraps the PyTorch model to a Sklearn model with Skorch library. """

    return SkorchModel(
        redefine_forward(model),
        criterion=trainer_config.criterion,
        optimizer=trainer_config.optimizer,
        lr=trainer_config.lr,
        max_epochs=trainer_config.epochs,
        batch_size=trainer_config.batch_size,
        train_split=None,
        callbacks="disable",
        device=trainer_config.device,
        criterion__reduce=False     # no reduction in order to use the sample weights for loss adjusting
    )


def redefine_forward(model: nn.Module) -> nn.Module:
    """
    Adds sample_weights parameter to the forward function of PyTorch model in order to allow using sample weights by
    wrapped Sklearn model:
    https://skorch.readthedocs.io/en/stable/user/FAQ.html#i-want-to-use-sample-weight-how-can-i-do-this
    """
    class TorchModelWithSampleWights(model.__class__):
        def forward(self, X, sample_weight=None):
            return super(TorchModelWithSampleWights, self).forward(X)

    return TorchModelWithSampleWights(model.linear.in_features, model.linear.out_features)
