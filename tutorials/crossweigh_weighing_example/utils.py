import logging
from typing import Union, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<PAD>", "<UNK>"]


def vocab_and_vectors(filename: str) -> (dict, np.ndarray):
    """
    Reads pretrained word embedding and builds 1) a matrix (words x embedding dim), 2) word to id dictionary
    :param filename: path to file with pretrained word embeddings
    :return: word2id, embedding matrix
    """
    with open(filename, encoding="UTF-8") as in_file:
        parts = in_file.readline().strip().split(" ")
        word_embedding_matrix = np.zeros(
            (int(parts[0]) + len(SPECIAL_TOKENS), int(parts[1]))
        )
        word2id = dict()
        for idx, token in enumerate(SPECIAL_TOKENS):
            word2id[token] = idx
        nextword_id = len(SPECIAL_TOKENS)
        for line in in_file:
            parts = line.strip().split(" ")
            word = parts[0]
            if word not in word2id:
                emb = [float(v) for v in parts[1:]]
                word_embedding_matrix[nextword_id] = emb
                word2id[word] = nextword_id
                nextword_id += 1
    return word2id, word_embedding_matrix


def get_data_features(
        input_data: pd.Series,
        word2id: dict,
        maxlen: int,
        samples_column: int,
        labels_column: int = None,
) -> Union[Tuple[torch.LongTensor, torch.LongTensor], torch.LongTensor]:
    """
    This function reads the input data saved as a DataFrame and encode sentences with words ids.
    :param path_train_data: path to .csv file with input data
    :param word2id: dictionary of words to their ids that corresponds to pretrained embeddings
    :param column_num: number of a column in DataFrame with input data where samples are stored
    :param maxlen: maximum length of encoded samples: if length of tokens > maxlen, reduce it to maxlen, else padding
    :return:
    """
    enc_input_samples = encode_samples(
        list(input_data.iloc[:, samples_column]), word2id, maxlen
    )
    # inputs_x_tensor = torch.LongTensor(enc_input_samples)
    # inputs_x_dataset = torch.utils.data.TensorDataset(inputs_x_tensor)

    if labels_column:
        labels_tensor = torch.LongTensor(list(input_data.iloc[:, labels_column]))
        # labels_dataset = torch.utils.data.TensorDataset(labels_tensor)
        return enc_input_samples, labels_tensor

    return enc_input_samples


# def get_dev_data(path_dev_feature_labels: str, word2id: dict, maxlen: int) -> TensorDataset:
#     """ Read dev data with gold labels and turn it into TensorDataset(features, labels)"""
#     dev_data = pd.read_csv(path_dev_feature_labels)
#     enc_dev_samples = encode_samples(list(dev_data.iloc[:, 1]), word2id, maxlen)
#     dev_samples_tensor = torch.LongTensor(enc_dev_samples)
#     dev_labels_tensor = torch.LongTensor(list(dev_data.iloc[:, 2]))
#
#     dev_feature_labels_dataset = torch.utils.data.TensorDataset(dev_samples_tensor, dev_labels_tensor)
#     return dev_feature_labels_dataset

