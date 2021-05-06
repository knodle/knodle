from knodle.trainer.config import TrainerConfig

from knodle.trainer.baseline.majority import MajorityVoteTrainer, MajorityConfig
from knodle.trainer.knn_denoising.knn import KnnDenoisingTrainer, KNNConfig
from knodle.trainer.snorkel.snorkel import SnorkelKNNDenoisingTrainer, SnorkelTrainer, SnorkelConfig, SnorkelKNNConfig

from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.auto_config import AutoConfig