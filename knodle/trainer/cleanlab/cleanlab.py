import numpy as np
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier
from torch.optim import SGD
from torch.utils.data import TensorDataset

from knodle.trainer import MajorityVoteTrainer
from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.cleanlab.config import CleanLabConfig
from knodle.transformation.majority import z_t_matrices_to_majority_vote_probs
from knodle.transformation.torch_input import input_to_2_dim_numpy


@AutoTrainer.register('cleanlab')
class CleanLabTrainer(MajorityVoteTrainer):
    def __init__(self, test_x, test_y, **kwargs):
        if kwargs.get("trainer_config", None) is None:
            kwargs["trainer_config"] = CleanLabConfig(optimizer=SGD, lr=0.001)
        super().__init__(**kwargs)

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ) -> None:
        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)

        # since CL accepts only sklearn.classifier compliant class, we wraps the PyTorch model
        self.model = NeuralNetClassifier(
            self.model,
            criterion=self.trainer_config.criterion,
            optimizer=self.trainer_config.optimizer,
            lr=self.trainer_config.lr,
            max_epochs=self.trainer_config.epochs,
            batch_size=self.trainer_config.batch_size,
            train_split=None,
            callbacks="disable",
            device=self.trainer_config.device
        )
        # turn input to the CL-compatible format
        model_input_x_numpy = input_to_2_dim_numpy(self.model_input_x)

        # calculate labels based on t and z
        noisy_y_train = np.argmax(
            z_t_matrices_to_majority_vote_probs(self.rule_matches_z, self.mapping_rules_labels_t), axis=1
        )

        # CL denoising and training
        rp = LearningWithNoisyLabels(
            clf=self.model, seed=self.trainer_config.seed,
            cv_n_folds=self.trainer_config.cv_n_folds,
            prune_method=self.trainer_config.prune_method,
            converge_latent_estimates=self.trainer_config.converge_latent_estimates,
            pulearning=self.trainer_config.pulearning,
            n_jobs=self.trainer_config.n_jobs
        )
        _ = rp.fit(model_input_x_numpy, noisy_y_train)
