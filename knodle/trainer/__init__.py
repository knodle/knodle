from knodle.trainer.config import TrainerConfig

from knodle.trainer.baseline.no_denoising import NoDenoisingTrainer
from knodle.trainer.knn_denoising.knn_denoising import KnnDenoisingTrainer
from knodle.trainer.snorkel.snorkel_trainer import SnorkelKNNDenoisingTrainer, SnorkelTrainer

from knodle.trainer.auto_trainer import AutoTrainer