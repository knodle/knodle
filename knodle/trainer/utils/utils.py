from snorkel.classification import cross_entropy_with_probs
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, KLDivLoss
from torch.optim import AdamW, SGD


def create_optimizer(
    model, optimizer: str, learning_rate: float = 0.1, epsilon: float = 0.01
):
    if optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)

    elif optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError("Just AdamW or SGD available")
    return optimizer


def create_criterion(criterion: str):
    if criterion is None:
        criterion = CrossEntropyLoss()
    elif criterion == "BCEWithLogitsLoss":
        criterion = BCEWithLogitsLoss()
    elif criterion == "KLDivLoss":
        criterion = KLDivLoss()
    elif criterion == "cross_entropy_with_probs":
        criterion = cross_entropy_with_probs
    elif criterion == "cross_entropy":
        criterion = CrossEntropyLoss()
    else:
        raise NotImplementedError(
            "You have chosen a loss function which is not implemented."
        )

    return criterion
