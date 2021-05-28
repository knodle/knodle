from typing import Callable

from knodle.trainer.config import TrainerConfig


class AutoConfig:
    """ The factory class for creating Config classes of training executors
    See See https://medium.com/@geoffreykoh/implementing-the-factory-
    pattern-via-dynamic-registry-and-python-decorators-479fc1537bbe
    """

    registry = {}
    """ Internal registry for available trainers """

    def __init__(self, name, **kwargs):
        self.config = self.create_config(name, **kwargs)

    @classmethod
    def create_config(cls, name: str, **kwargs) -> TrainerConfig:

        exec_class = cls.registry[name]
        executor = exec_class(**kwargs)
        return executor

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: TrainerConfig) -> Callable:
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper
