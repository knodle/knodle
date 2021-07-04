from knodle.trainer.config import TrainerConfig

from knodle.trainer.baseline.majority import MajorityVoteTrainer, MajorityConfig
from knodle.trainer.knn_aggregation.knn import kNNAggregationTrainer, kNNConfig
from knodle.trainer.snorkel.snorkel import SnorkelkNNAggregationTrainer, SnorkelTrainer, SnorkelConfig, SnorkelkNNConfig

from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.auto_config import AutoConfig