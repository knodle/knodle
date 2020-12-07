import numpy as np
from knodle.model.BidirectionalLSTM.BidirectionalLSTM import BidirectionalLSTM
from knodle.trainer.crossweight_denoising import utils
from knodle.trainer.crossweight_denoising.CrossWeightTrainer import CrossWeightTrainer

NUM_CLASSES = 38


def train_crossweight(path_x, path_z, path_t, path_word_emb_file):
    word2id, id2word, word_embedding_matrix = utils.vocab_and_vectors(path_word_emb_file, ['<PAD>', '<UNK>'])
    # todo: now glove embeddings are read from the file once again in crossweight training --> one should leave...

    x = np.load(path_x)
    z = np.load(path_z)
    t = np.load(path_t)

    model = BidirectionalLSTM(word_embedding_matrix.shape[0],
                              word_embedding_matrix.shape[1],
                              word_embedding_matrix,
                              NUM_CLASSES)

    trainer = CrossWeightTrainer(model)
    trainer.train(x, z, t, path_word_emb_file)          # todo: optimize import so that only three matrices


if __name__ == "__main__":
    train_crossweight("data_for_testing/x_matrix.npy",
                      "data_for_testing/z_matrix.npy",
                      "data_for_testing/t_matrix.npy",
                      "data/glove.840B.300d.txt.filtered")
    # todo: optimize import so that only three matrices


