from abc import abstractmethod

from .config import LabelerConfig


class Labeler:
    def __init__(
            self,
            labeler_config: LabelerConfig = None,
    ):

        if labeler_config is None:
            self.labeler_config = LabelerConfig()
        else:
            self.labeler_config = labeler_config

    @abstractmethod
    def label(self):
        pass
