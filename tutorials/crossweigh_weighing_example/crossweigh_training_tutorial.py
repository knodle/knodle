import argparse
import os
import sys
import torch
from joblib import load

from knodle.model.bidirectional_lstm_model import BidirectionalLSTM
from knodle.trainer.config.crossweigh_denoising_config import CrossWeighDenoisingConfig
from knodle.trainer.config.crossweigh_trainer_config import TrainerConfig
from knodle.trainer.crossweigh_weighing.crossweigh import CrossWeigh
from tutorials.crossweigh_weighing_example import utils

NUM_CLASSES = 43
CLASS_WEIGHTS = torch.FloatTensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])


def train_crossweigh(
        path_t: str, path_train_samples: str, path_z: str, path_dev_features_labels: str, path_word_emb_file: str
) -> None:
    """
    Training the model with CrossWeigh model denoising
    :param path_train_samples: path to DataFrame with training data
    :param path_z: path to binary matrix that contains info about rules matched in samples (samples x rules)
    :param path_t: path to binary matrix that contains info about which rule corresponds to which label (rule x labels)
    :param path_dev_features_labels: path to DataFrame with development data (1st column - samples, 2nd column - labels)
    :param path_word_emb_file: path to file with pretrained embeddings
    """

    word2id, word_embedding_matrix = utils.vocab_and_vectors(path_word_emb_file)

    rule_matches_z = load(path_z)
    rule_assignments_t = load(path_t)
    train_input_x = utils.get_train_features(path_train_samples, word2id)

    dev_features_labels_dataset = utils.get_dev_data(path_dev_features_labels, word2id)

    model = BidirectionalLSTM(word_embedding_matrix.shape[0],
                              word_embedding_matrix.shape[1],
                              word_embedding_matrix,
                              NUM_CLASSES)

    trainer = CrossWeigh(model=model,
                         rule_assignments_t=rule_assignments_t,
                         inputs_x=train_input_x,
                         rule_matches_z=rule_matches_z,
                         dev_features_labels=dev_features_labels_dataset,
                         trainer_config=TrainerConfig(model=model,
                                                      class_weights=CLASS_WEIGHTS,
                                                      output_classes=NUM_CLASSES),
                         denoising_config=CrossWeighDenoisingConfig(model=model,
                                                                    class_weights=CLASS_WEIGHTS,
                                                                    output_classes=NUM_CLASSES))
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--rule_assignments_t", help="")
    parser.add_argument("--path_train_samples", help="")
    parser.add_argument("--rule_matches_z", help="")
    parser.add_argument("--dev_features_labels", help="")
    parser.add_argument("--word_embeddings", help="")

    args = parser.parse_args()

    train_crossweigh(args.rule_assignments_t,
                     args.path_train_samples,
                     args.rule_matches_z,
                     args.dev_features_labels,
                     args.word_embeddings)
