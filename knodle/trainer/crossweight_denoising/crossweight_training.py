import torch
import numpy as np
from knodle.model.BidirectionalLSTM.BidirectionalLSTM import BidirectionalLSTM
from knodle.trainer.config.CrossWeightDenoisingConfig import CrossWeightDenoisingConfig
from knodle.trainer.crossweight_denoising import utils
from knodle.trainer.crossweight_denoising.CrossWeightTrainer import CrossWeightTrainer

NUM_CLASSES = 39
CLASS_WEIGHTS = torch.FloatTensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0])


def train_crossweight(path_x: str, path_z: str, path_t: str, path_word_emb_file: str):
    word2id, id2word, word_embedding_matrix = utils.vocab_and_vectors(path_word_emb_file, ['<PAD>', '<UNK>'])

    inputs_x = np.load(path_x)
    rule_matches_z = np.load(path_z)
    rule_assignments_t = np.load(path_t)

    model = BidirectionalLSTM(word_embedding_matrix.shape[0],
                              word_embedding_matrix.shape[1],
                              word_embedding_matrix,
                              NUM_CLASSES)

    trainer = CrossWeightTrainer(model, CrossWeightDenoisingConfig(model=model,
                                                                   class_weights=CLASS_WEIGHTS,
                                                                   output_classes=NUM_CLASSES))
    trainer.train(inputs_x, rule_matches_z, rule_assignments_t)


if __name__ == "__main__":
    train_crossweight("data_for_testing/x_matrix.npy",
                      "data_for_testing/z_matrix.npy",
                      "data_for_testing/t_matrix.npy",
                      "data/glove.840B.300d.txt.filtered")
    # todo: optimize import so that only three matrices


