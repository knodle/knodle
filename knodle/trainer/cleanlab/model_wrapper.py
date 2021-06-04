from skorch import NeuralNetClassifier

from knodle.trainer import TrainerConfig


def wrap_model(model, trainer_config: TrainerConfig):
    """ The function wraps the PyTorch model to a Sklearn model. """

    # criterion should have no reduction in order to use the sample weights for loss adjusting
    # trainer_config.criterion.reduction = "none"

    # model = redefine_forward(model)

    return NeuralNetClassifier(
        model,
        # criterion=trainer_config.criterion(reduction="none"),
        criterion=trainer_config.criterion,
        optimizer=trainer_config.optimizer,
        lr=trainer_config.lr,
        max_epochs=trainer_config.epochs,
        batch_size=trainer_config.batch_size,
        train_split=None,
        callbacks="disable",
        device=trainer_config.device,

        # criterion__reduce = False,

    )
#
#
#
# # [in the plugin file]
# from code import Model, instance
#
#
#
# newmodel = MyModel(a="a name", b="some other stuff")
# instance.register(newmodel)
#
#
# def redefine_forward(model):
