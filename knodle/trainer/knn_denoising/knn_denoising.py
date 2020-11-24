import logging
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from torch import Tensor
from torch.nn import Module
import numpy as np
from tqdm import tqdm
from knodle.final_label_decider.FinalLabelDecider import get_majority_vote_probabilities
from knodle.trainer.config.TrainerConfig import TrainerConfig
from knodle.trainer.ds_model_trainer.ds_model_trainer import DsModelTrainer
from knodle.trainer.utils.utils import print_section

logger = logging.getLogger(__name__)

TFIDF_VALUES_PARAMETER_NAME = "tfidf_values"
K_PARAMETER_NAME = "k"


class KnnDenoising(DsModelTrainer):
    def __init__(self, model: Module, trainer_config: TrainerConfig = None):
        super().__init__(model, trainer_config)

    def train(self, inputs: Tensor, rule_matches: np.ndarray, epochs, **kwargs):
        """
        This function first denoises the labeling functions with kNN, gets final labels with majoriyt vote and trains
        finally the model.
        Args:
            inputs: Input tensors. These tensors will be fed to the provided model (instaces x features)
            rule_matches: All rule matches (instances x rules)
            tfidf_values: Tftdf values of input texts (instances x tfidf_features)
            epochs: Epochs to train
            k: Number of neighbors to consider in kNN approach.
        """
        self.check_if_kwargs_provided(kwargs)
        k, tfidf_values = self.extract_arguments_from_kwargs(kwargs)

        assert (
            inputs.shape == tfidf_values.shape
        ), "Shapes of inputs and tfidf values have to be the same"
        assert len(inputs) == len(
            rule_matches
        ), "Shapes of inputs and applied labeling functions have to be the same"

        assert epochs > 0, "Epochs has to be set with a positive number"
        assert k > 0, "k has to be set with a positive number"

        denoised_applied_lfs = self.denoise_rule_matches(
            rule_matches, tfidf_values=tfidf_values, k=k
        )
        labels = get_majority_vote_probabilities(
            rule_matches=denoised_applied_lfs,
            output_classes=self.trainer_config.output_classes,
        )
        labels = Tensor(labels)

        self.model.train()

        print_section("Training starts", logger)

        for current_epoch in range(epochs):
            logger.info("Epoch: {}".format(current_epoch))
            self.model.zero_grad()
            predictions = self.model(inputs)
            loss = self.trainer_config.criterion(predictions, labels)
            logger.info("Loss is: {}".format(loss))
            loss.backward()
            self.trainer_config.optimizer.step()

        print_section("Training done", logger)

    def denoise_rule_matches(self, rule_matches: np.ndarray, **kwargs) -> np.ndarray:
        """
        Denoises the applied weak supervision source.
        Args:
            rule_matches: Matrix with all applied weak supervision sources. Shape: (Instances x Rules)
            tfidf_values: Text in tfidf values. Shape: (Instances x Features)
            k: How many neighbors to consider. Hyperparameter

        Returns: Denoised / Improved applied labeling function matrix. Shape: (Instances x Rules)

        """
        k, tfidf_values = self.extract_arguments_from_kwargs(kwargs)
        logger.info("Start denoising labeling functions with k: {}.".format(k))

        if k == 1:
            return rule_matches

        logger.info("This can take a while ...")

        neighbors = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(tfidf_values)
        distances, indices = neighbors.kneighbors(tfidf_values)
        new_lfs = self.activate_all_neighbors(rule_matches, indices)
        return new_lfs

    def activate_all_neighbors(
        self, lfs: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        """
        Find all closest neighbors and take the same label ids
        Args:
            lfs:
            indices:

        Returns:

        """
        new_lfs_array = np.full(lfs.shape, fill_value=-1)

        for index, lf in tqdm(enumerate(lfs)):

            try:
                matched_lfs = np.where(lf != -1)[0]
                if len(matched_lfs) == 0:
                    continue
                matched_lfs = matched_lfs[:, np.newaxis]
                neighbors = indices[index]
                to_replace = new_lfs_array[neighbors, matched_lfs]
                label_matched_lfs = lf[matched_lfs][:, 0]
                tiled_labels = np.tile(
                    np.array(label_matched_lfs), (to_replace.shape[1], 1)
                ).transpose()
                new_lfs_array[neighbors, matched_lfs] = tiled_labels
            except IndexError:
                # IndexError can occur in some special cases
                pass

        return new_lfs_array

    def check_if_kwargs_provided(self, kwargs) -> None:
        assert TFIDF_VALUES_PARAMETER_NAME in kwargs.keys(), (
            "tfidf_values are needed. Please provide them when " "calling train"
        )
        assert (
            K_PARAMETER_NAME in kwargs.keys()
        ), "k is needed. Please provide it when calling train"

    def extract_arguments_from_kwargs(self, kwargs) -> (int, csr_matrix):
        return kwargs[K_PARAMETER_NAME], kwargs[TFIDF_VALUES_PARAMETER_NAME]
