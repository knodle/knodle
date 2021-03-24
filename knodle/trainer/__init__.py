from knodle.trainer.config import TrainerConfig

from knodle.trainer.baseline.majority import MajorityVoteTrainer
from knodle.trainer.knn_denoising.knn import KnnDenoisingTrainer
from knodle.trainer.snorkel.snorkel import SnorkelKNNDenoisingTrainer, SnorkelTrainer

from knodle.trainer.auto_trainer import AutoTrainer