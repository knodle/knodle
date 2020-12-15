import numpy as np
import re
import itertools
from typing import Union
import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from torch.nn import Module

ARG1 = "$ARG1"
ARG2 = "$ARG2"

disable_cuda = True
device = None
if not disable_cuda and torch.cuda.is_available():
    print("Using GPU")
    device = torch.device('cuda')
else:
    print("Using CPU")
    device = torch.device('cpu')


def vocab_and_vectors(filename: str, special_tokens: list) -> (dict, dict, np.ndarray):
    """special tokens have all-zero word vectors"""
    with open(filename, encoding="UTF-8") as in_file:
        parts = in_file.readline().strip().split(" ")
        num_vecs = int(parts[0]) + len(special_tokens)  # + 1
        dim = int(parts[1])

        matrix = np.zeros((num_vecs, dim))
        word_to_id, id_to_word = dict(), dict()

        nextword_id = 0
        for token in special_tokens:
            word_to_id[token] = nextword_id
            id_to_word[nextword_id] = token
            nextword_id += 1

        for line in in_file:
            parts = line.strip().split(' ')
            word = parts[0]
            if word not in word_to_id:
                emb = [float(v) for v in parts[1:]]
                matrix[nextword_id] = emb
                word_to_id[word] = nextword_id
                id_to_word[nextword_id] = word
                nextword_id += 1
    return word_to_id, id_to_word, matrix


def add_padding(tokens: list, maxlen: int) -> list:
    """ Provide padding of the encoded tokens to the maxlen; if length of tokens > maxlen, reduce it to maxlen """
    padded_tokens = [0] * maxlen
    for i in range(0, min(len(tokens), maxlen)):
        padded_tokens[i] = tokens[i]
    return padded_tokens


def get_id(item: Union[int, str], dic: dict) -> Union[int, str]:
    """ This function checks if there is a key in a dict. If so, it returns a value of a dict.
    Creates a dictionary pair {key : max(dict value) + 1} otherwise """
    if item in dic:
        item_id = dic[item]
    else:
        item_id = len(dic)
        dic[item] = item_id
    return item_id


def update_dict(key: Union[int, str], value: Union[int, str], dic: dict) -> None:
    """ This function checks if there is a key in a dict. If so, it added new value to the list of values for this key.
    Create a {key: {value}} pair otherwise."""
    if key in dic:
        dic[key].append(value)
    else:
        dic[key] = [value]


def convert_pattern_to_regex(pattern: str) -> str:
    """ Transfor the pattern into regex by char escaping and adding of optional articles in front of the arguments"""
    return re.sub("\\\\\\$ARG", "(A )?(a )?(The )?(the )?\\$ARG", re.escape(pattern))


def get_match_matrix_row(num_columns: int, matched_columns=None) -> list:
    """
    There is a function that creates a list which will be transformed into matrix row in future. All elements in the
    liste qual zeros, apart from the ones with indices given in matched_columns list - they equal ones.
    :param num_columns: length of created list = future row vector 2nd dimensionality
    :param matched_columns: list of element indices that equal to 1
    :return: binary list
    """
    matrix_row = [0] * num_columns
    if matched_columns:
        for m_column in matched_columns:
            matrix_row[m_column] = 1
    return matrix_row


def get_extracted_sample(sample: str) -> list:
    """ Using the Spacy entity recognition information, forms the subtractions of the samples in form of
    "ARG1 <some words> ARG2" strings which will be used to look for patterns in them """
    return [(ARG1 + sample["text"][ent1["end"]:ent2["start"]] + ARG2) if ent1["end"] < ent2["end"]
            else (ARG2 + sample["text"][ent2["end"]:ent1["start"]] + ARG1)
            for ent1, ent2 in itertools.permutations(sample["ents"], 2)]


def initialize_weights(model: Module) -> None:
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.xavier_uniform_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


def convert2tensor(samples: np.ndarray,
                   labels: np.ndarray,
                   idx: np.ndarray,
                   batch_size: int,
                   shuffle: bool = True
                   ) -> DataLoader:

    tensor_words = torch.LongTensor(samples).to(device=device)
    tensor_target = torch.LongTensor(labels).to(device=device)
    tensor_idx = torch.LongTensor(idx).to(device=device)
    dataset = torch.utils.data.TensorDataset(tensor_words, tensor_target, tensor_idx)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def save_sample_weights(sample_weights, path_to_weights_dir):
    try:
        os.mkdir(path_to_weights_dir)
        print("Directory ", path_to_weights_dir, " Created ")
    except FileExistsError:
        print("Directory ", path_to_weights_dir, " already exists")
    np.save(os.path.join(path_to_weights_dir, "sample_weight.npy"), sample_weights)


def get_shuffled_idx(matrix: np.ndarray) -> np.ndarray:
    """
    Get shuffled row indices of dataset
    :param matrix: numpy array where we want to shuffle rows
    :return: numpy array of shuffled indices
    """
    return np.random.rand(matrix.shape[0]).argsort()
