import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from knodle.model.bidirectional_lstm_model import BidirectionalLSTM

from knodle.trainer.config.crossweight_trainer_config import TrainerConfig
from knodle.trainer.config.crossweight_denoising_config import CrossWeightDenoisingConfig

from knodle.trainer.crossweight_weighing import utils
from knodle.trainer.crossweight_weighing.crossweight import CrossWeight

logger = logging.getLogger(__name__)

NUM_CLASSES = 39
CLASS_WEIGHTS = torch.FloatTensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0])


def train_crossweight(
        path_t: str, path_z: str, path_train_data: str, path_dev_samples: str, path_dev_labels: str,
        path_word_emb_file: str) -> None:
    """
    Training the model with CrossWeight model denoising
    :param path_train_data: path to matrix with training data (DataFrame with row samples or Numpy array with encoded)
    :param path_z: path to binary matrix that contains info about rules matched in samples (samples x rules)
    :param path_t: path to binary matrix that contains info about which rule correspond to which label
    :param path_dev_samples: path to data using for development
    :param path_dev_labels: path to labels using for development
    :param path_word_emb_file: path to file with pretrained embeddings
    """

    word2id, word_embedding_matrix = utils.vocab_and_vectors(path_word_emb_file, ['<PAD>', '<UNK>'])

    rule_matches_z = np.load(path_z)
    rule_assignments_t = np.load(path_t)
    train_input_x = get_train_input_x(path_train_data, word2id)

    dev_samples, dv_labels = read_dev_data(path_dev_samples, path_dev_labels)

    model = BidirectionalLSTM(word_embedding_matrix.shape[0],
                              word_embedding_matrix.shape[1],
                              word_embedding_matrix,
                              NUM_CLASSES)

    trainer = CrossWeight(model=model,
                          rule_assignments_t=rule_assignments_t,
                          inputs_x=train_input_x,
                          rule_matches_z=rule_matches_z,
                          dev_inputs=dev_samples,
                          dev_labels=dv_labels,
                          trainer_config=TrainerConfig(model=model,
                                                              class_weights=CLASS_WEIGHTS,
                                                              output_classes=NUM_CLASSES),
                          denoising_config=CrossWeightDenoisingConfig(model=model,
                                                                             class_weights=CLASS_WEIGHTS,
                                                                             output_classes=NUM_CLASSES))
    trainer.train()


def get_train_input_x(path_train_data: str, word2id: dict) -> TensorDataset:
    """

    :param path_train_data: path to file with input data. If .npy with encoded samples, return as Dataloader.
    If .csv with row samples, do encoding and convert to Dataloader.
    :param word2id: dictionary of words to their ids that corresponds to pretrained embeddings.
    :return: encoded training set as dataset
    """
    if path_train_data.endswith(".csv"):
        logger.info("Train samples are passed as .csv file. The data will be loaded from the dataframe.")
        train_inputs_x = encode_x(path_train_data, word2id)
        return train_inputs_x
    elif path_train_data.endswith(".npy"):
        logger.info("Train samples are passed as .npy file. The data will be loaded from the numpy matrix.")
        train_inputs_x = torch.utils.data.TensorDataset(torch.Tensor(np.load(path_train_data)))
        return train_inputs_x
    else:
        raise ValueError("Wrong train data format! It should be either stored as a dataframe or as a numpy matrix.")


def encode_x(path_train_data: str, word2id: dict, maxlen: int = 50) -> TensorDataset:
    """
    This function reads the input data saved as a DataFrame and encode sentences with words ids.
    :param path_train_data: path to .csv file with input data
    :param word2id: dictionary of words to their ids that corresponds to pretrained embeddings
    :param maxlen: maximal length of sentences. Sentences longer are cut, sentences shorter are padded with 0.
    :return:
    """
    input_data = pd.read_csv(path_train_data)
    input_samples = list(input_data["sample"])
    enc_input_samples = []

    for sample in input_samples:
        enc_tokens = [word2id.get(token, 1) for token in sample.lstrip().split(" ")]
        enc_input_samples.append(np.asarray(utils.add_padding(enc_tokens, maxlen), dtype="float64"))

    inputs_x_tensor = torch.Tensor(enc_input_samples)
    inputs_x_dataset = torch.utils.data.TensorDataset(inputs_x_tensor)

    return inputs_x_dataset


def read_dev_data(path_dev_samples: str, path_dev_labels: str) -> (TensorDataset, np.ndarray):
    """ Read dev data with gold labels """
    dev_samples = np.load(path_dev_samples)
    dev_sample_tensor = torch.Tensor(dev_samples)
    dev_samples = torch.utils.data.TensorDataset(dev_sample_tensor)
    dv_labels = np.load(path_dev_labels)
    return dev_samples, dv_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]))
    parser.add_argument("--rule_assignments_t", help="")
    parser.add_argument("--rule_matches_z", help="")
    parser.add_argument("--path_train_data", help="")
    parser.add_argument("--dev_samples", help="")
    parser.add_argument("--dev_labels", help="")
    parser.add_argument("--word_embeddings", help="")

    args = parser.parse_args()

    train_crossweight(args.rule_assignments_t,
                      args.rule_matches_z,
                      args.path_train_data,
                      args.dev_samples,
                      args.dev_labels,
                      args.word_embeddings)
