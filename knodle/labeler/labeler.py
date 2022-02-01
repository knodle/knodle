from abc import abstractmethod

from .config import LabelerConfig


class Labeler:
    def __init__(
            self,
            labeler_config: LabelerConfig = None,
    ):
        self.labeler_config = LabelerConfig() if labeler_config is None else labeler_config

    @abstractmethod
    def label(self):
        pass
