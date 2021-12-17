import os

from knodle.trainer.baseline.config import BaseTrainerConfig
from knodle.trainer.auto_config import AutoConfig


@AutoConfig.register("cosine")
class CosineConfig(BaseTrainerConfig):
    def __init__(
            self,
            args,
            **kwargs
    ):

        super().__init__(**kwargs)
        self.metric = args.metric
        self.T1 = args.T1
        self.T2 = args.T2
        self.T3 = args.T3
        self.bert_backbone = args.bert_backbone
        self.max_sen_len = args.max_sen_len
        self.bert_dropout_rate = args.bert_dropout_rate
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.train_eval_freq = args.train_eval_freq
        self.eval_freq = args.eval_freq
        self.warmup_steps = args.warmup_steps
        self.lr = args.lr
        self.max_grad_norm = args.max_grad_norm
        self.self_training_eps = args.self_training_eps
        self.distmetric = args.distmetric
        self.self_training_confreg = args.self_training_confreg
        self.self_training_power = args.self_training_power
        self.self_training_contrastive_weight = args.self_training_contrastive_weight
        self.use_cuda = args.use_cuda
        self.seed = args.manualSeed
        self.weight_decay = args.weight_decay
        self.soft = True if args.teacher_label_type else False


    # def get_cache_file(self):
    #     nn_type = "ann" if self.use_approximation else "knn"
    #     file_tags = f"{self.k}_{nn_type}"
    #     if self.caching_suffix:
    #         file_tags += f"_{self.caching_suffix}"
    #
    #     cache_file = os.path.join(
    #         self.caching_folder,
    #         f"denoised_rule_matches_z_{file_tags}.lib"
    #     )
    #
    #     return cache_file
